"""
研究趋势分析主流程

独立于每日研究模式（modes/daily_research.py），实现完整的研究趋势分析流水线：
1. 按关键词 + 时间范围从 ArXiv 搜索论文
2. 为每篇论文生成 LLM TLDR（无评分）
3. 使用 Skills 系统进行整体趋势分析
4. 生成 Markdown + HTML 报告
5. 发送通知
"""

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

from config import settings
from utils.logger import setup_logger
from utils.token_counter import token_counter
from sources.arxiv_source import ArxivSource
from sources.openalex_source import OpenAlexSource, JOURNAL_ISSN_MAP
from agents.trend_agent import TrendAgent
from report.trend.reporter import TrendReporter
from notifications import NotifierAgent

logger = setup_logger("TrendResearch")


def _keywords_hash(keywords: List[str]) -> str:
    """对关键词集合做规范化哈希（顺序无关、大小写无关），用于 cache 失效判定。"""
    normalized = sorted(kw.strip().lower() for kw in keywords if kw and kw.strip())
    joined = "\n".join(normalized)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


class _ScoreCache:
    """
    论文相关性评分本地缓存。

    存储格式（JSON）:
        {
          "keywords_hash": "<md5>",
          "scores": {"<paper_id>": <float>, ...}
        }

    当 keywords_hash 与当前关键词集合不一致时，整个 cache 视为失效（返回空），
    下次写入时会覆盖旧数据。保证用户修改关键词后旧评分不会污染结果。
    """

    def __init__(self, path: Path, current_keywords_hash: str):
        self.path = path
        self.current_hash = current_keywords_hash
        self.scores: Dict[str, float] = {}
        self._loaded_hash: Optional[str] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.path.exists():
            logger.info(f"  [cache] 评分缓存不存在，将新建: {self.path}")
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._loaded_hash = data.get("keywords_hash")
            raw_scores = data.get("scores", {}) or {}
            if self._loaded_hash == self.current_hash:
                self.scores = {
                    k: float(v) for k, v in raw_scores.items() if isinstance(v, (int, float))
                }
                logger.info(f"  [cache] 命中关键词哈希，载入 {len(self.scores)} 条评分")
            else:
                logger.warning(
                    f"  [cache] 关键词哈希不匹配（old={self._loaded_hash}, new={self.current_hash}），"
                    f"旧评分视为失效，本次重新计算"
                )
        except Exception as e:
            logger.warning(f"  [cache] 读取评分缓存失败，忽略: {e}")

    def get(self, paper_id: str) -> Optional[float]:
        return self.scores.get(paper_id)

    def set(self, paper_id: str, score: float) -> None:
        with self._lock:
            self.scores[paper_id] = float(score)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"keywords_hash": self.current_hash, "scores": self.scores}
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp_path.replace(self.path)
            logger.info(f"  [cache] 评分缓存已写入（共 {len(self.scores)} 条）: {self.path}")
        except Exception as e:
            logger.warning(f"  [cache] 写入评分缓存失败: {e}")


class TrendResearchPipeline:
    """
    研究趋势分析流水线。

    参数:
        settings: 全局配置对象
        keywords: 搜索关键词列表
        date_from: 搜索起始日期
        date_to: 搜索截至日期
        sort_order: 排序方向 (ascending / descending)
        max_results: 最大论文数
    """

    def __init__(
        self,
        settings,
        keywords: List[str],
        date_from: date,
        date_to: date,
        sort_order: str = "ascending",
        max_results: int = 500,
        categories: List[str] = None,
        history_dir: Optional[Path] = None,
        dedupe_history: bool = False,
        match_mode: str = "AND",
        final_top_n: Optional[int] = None,
        score_pool_size: Optional[int] = None,
        enabled_sources: Optional[List[str]] = None,
        journals: Optional[List[str]] = None,
        max_results_per_source: Optional[Dict[str, int]] = None,
        openalex_email: Optional[str] = None,
        openalex_api_key: Optional[str] = None,
    ):
        self.settings = settings
        self.keywords = keywords
        self.date_from = date_from
        self.date_to = date_to
        self.sort_order = sort_order
        self.max_results = max_results
        self.categories = categories or []
        self.history_dir = history_dir or self.settings.HISTORY_DIR
        self.dedupe_history = dedupe_history
        self.match_mode = (match_mode or "AND").strip().upper()
        if self.match_mode not in {"AND", "OR"}:
            self.match_mode = "AND"
        # 当 final_top_n 设置且候选多于该值时，按关键词相关性本地评分后截断到 final_top_n
        self.final_top_n = final_top_n if (final_top_n and final_top_n > 0) else None
        # 打分候选池上限：历史过滤后若仍超过该值，只对前 N 篇（按时间倒序）做 LLM 打分，
        # 用于控制单次运行的 LLM 调用成本。None 表示不做二次截断。
        self.score_pool_size = score_pool_size if (score_pool_size and score_pool_size > 0) else None
        # 记录最近一次重排序的 paper_id -> score 映射，供通知阶段展示 Top-N 详情
        self._last_scores: Dict[str, float] = {}
        # topic discovery 可选复用 daily 的多源配置；默认保持 arXiv-only 兼容行为
        self.enabled_sources = enabled_sources or ["arxiv"]
        self.journals = journals or []
        self.max_results_per_source = max_results_per_source or {}
        self.openalex_email = openalex_email if openalex_email is not None else settings.OPENALEX_EMAIL
        self.openalex_api_key = (
            openalex_api_key if openalex_api_key is not None else settings.OPENALEX_API_KEY
        )

    def run(self):
        """执行研究趋势分析完整流程"""
        try:
            print("\n" + "=" * 80)
            print("🔬 研究趋势分析模式启动")
            print("=" * 80 + "\n")

            logger.info("=" * 80)
            logger.info("启动研究趋势分析模式")
            logger.info(f"  关键词: {self.keywords}")
            logger.info(f"  时间范围: {self.date_from} ~ {self.date_to}")
            logger.info(f"  排序方式: {self.sort_order}")
            logger.info(f"  最大结果数: {self.max_results}")
            if self.categories:
                logger.info(f"  ArXiv 分类: {self.categories}")
            logger.info(
                f"  历史去重: {'开启' if self.dedupe_history else '关闭'} "
                f"(history_dir={self.history_dir})"
            )
            logger.info(f"  关键词匹配模式: {self.match_mode}")
            if self.final_top_n:
                logger.info(f"  候选池→相关性截断 Top-N: {self.final_top_n}")
            logger.info("=" * 80)

            if settings.TOKEN_TRACKING_ENABLED:
                token_counter.reset()

            # ==================== 阶段1: 搜索论文 ====================
            logger.info(">>> 阶段1: 搜索候选论文...")

            # 启用 final_top_n 时延后标记历史：先取候选池、本地重排序后仅把最终 Top-N 写入历史，
            # 避免未入选的候选被误标记导致次日候选池枯竭。
            defer_history_mark = bool(self.dedupe_history and self.final_top_n)

            papers, history_sources = self._fetch_candidate_papers(defer_history_mark=defer_history_mark)

            # 不延后时，按源立即标记全部候选为已处理
            # arXiv-only 且 mark_after_fetch=True 时，ArxivSource 内部已完成历史写入，避免重复写文件
            should_mark_in_outer = not (
                set(history_sources.keys()) == {"arxiv"} and "openalex" not in history_sources
            )
            if self.dedupe_history and not defer_history_mark and papers and should_mark_in_outer:
                self._mark_papers_history(papers, history_sources)
                logger.info(f"  已将候选 {len(papers)} 篇论文写入历史")

            if not papers:
                logger.info("未搜索到任何论文。")
                print("\n未搜索到任何论文，程序退出。")
                self._send_result_notification(total_papers=0, report_paths={}, success=True)
                return

            logger.info(f"搜索到 {len(papers)} 篇论文")
            print(f"  搜索到 {len(papers)} 篇论文")

            # ==================== 阶段1.5（可选）: 本地相关性评分 + 截断 Top-N ====================
            # 用于 topic-discovery：候选池较大时按关键词相关性打分排序，仅保留 Top-N 进入后续处理
            if self.final_top_n and len(papers) > self.final_top_n:
                # 打分前先按 score_pool_size 控制单次 LLM 调用量（保留最前面的 N 篇）
                if self.score_pool_size and len(papers) > self.score_pool_size:
                    logger.info(
                        f">>> 阶段1.5a: 打分前按 score_pool_size={self.score_pool_size} 截断"
                        f"（{len(papers)} → {self.score_pool_size}）"
                    )
                    papers = papers[: self.score_pool_size]

                logger.info(
                    f">>> 阶段1.5: 本地相关性评分（{len(papers)} 候选 → Top {self.final_top_n}）..."
                )
                papers = self._rerank_and_truncate(papers, self.final_top_n)
                logger.info(f"  截断后保留 {len(papers)} 篇论文")
                print(f"  按相关性截断至 {len(papers)} 篇")

            # 若此前延后了历史标记，现在只把最终入选论文写入历史，
            # 未入选候选留待下次运行重新评分
            if defer_history_mark and papers:
                self._mark_papers_history(papers, history_sources)
                logger.info(f"  已将入选 {len(papers)} 篇论文写入历史")

            # ==================== 阶段2: 生成 TLDR ====================
            tldrs: Dict[str, str] = {}
            trend_agent = TrendAgent()

            if self.settings.RESEARCH_GENERATE_TLDR:
                logger.info(">>> 阶段2: 生成论文 TLDR...")

                if self.settings.ENABLE_CONCURRENCY and len(papers) > 1:
                    tldrs = self._generate_tldrs_concurrent(trend_agent, papers)
                else:
                    tldrs = self._generate_tldrs_sequential(trend_agent, papers)
            else:
                logger.info(">>> 阶段2: 跳过 TLDR 生成（配置关闭）")

            tldr_count = sum(1 for v in tldrs.values() if v)
            if self.settings.RESEARCH_GENERATE_TLDR:
                logger.info(f"  TLDR 生成完成: {tldr_count}/{len(papers)} 篇成功")

            # ==================== 阶段3: 趋势分析 ====================
            logger.info(">>> 阶段3: 执行趋势分析...")
            print(f"  执行趋势分析 ({len(self.settings.RESEARCH_ENABLED_SKILLS)} 个技能)...")

            trend_analysis = trend_agent.analyze_trends(
                keywords=self.keywords,
                papers=papers,
                date_from=self.date_from,
                date_to=self.date_to,
                tldrs=tldrs,
            )

            analysis_count = len(trend_analysis)
            logger.info(f"  趋势分析完成: {analysis_count} 个技能产生了结果")

            # ==================== 阶段4: 生成报告 ====================
            logger.info(">>> 阶段4: 生成研究趋势报告...")

            reporter = TrendReporter()
            report_paths = reporter.render(
                papers=papers,
                tldrs=tldrs,
                trend_analysis=trend_analysis,
                keywords=self.keywords,
                date_from=self.date_from,
                date_to=self.date_to,
                sort_order=self.sort_order,
                token_usage=(
                    token_counter.get_summary() if settings.TOKEN_TRACKING_ENABLED else None
                ),
            )

            # ==================== 阶段5: 发送通知 ====================
            logger.info(">>> 阶段5: 发送通知...")

            # 构建 Top-N 论文摘要：最终入选的论文列表（已在重排序后截断），
            # 附带本轮打分 + TLDR + 原文链接，方便在 Telegram / 邮件中直接跳转。
            notification_top_n = self.final_top_n or self.settings.NOTIFICATION_TOP_N
            top_papers_payload: List[Dict[str, Any]] = []
            for p in papers[: notification_top_n or len(papers)]:
                top_papers_payload.append(
                    {
                        "title": p.title,
                        "score": self._last_scores.get(p.paper_id, 0.0),
                        "source": getattr(p, "source", "unknown"),
                        "tldr": tldrs.get(p.paper_id, "") if tldrs else "",
                        "url": getattr(p, "url", "") or "",
                    }
                )

            self._send_result_notification(
                total_papers=len(papers),
                report_paths=report_paths,
                success=True,
                trend_skills_count=analysis_count,
                tldr_count=tldr_count,
                token_usage=(
                    token_counter.get_summary() if settings.TOKEN_TRACKING_ENABLED else None
                ),
                top_papers=top_papers_payload,
            )

            # ==================== 完成 ====================
            logger.info("=" * 80)
            logger.info("✅ 研究趋势分析完成！")
            logger.info("=" * 80)

            print("\n" + "=" * 80)
            print("🎉 研究趋势分析完成！")
            print("=" * 80)
            print("📊 统计信息:")
            print(f"   • 关键词: {', '.join(self.keywords)}")
            print(f"   • 时间范围: {self.date_from} ~ {self.date_to}")
            if self.categories:
                print(f"   • ArXiv 分类: {', '.join(self.categories)}")
            print(f"   • 搜索到论文: {len(papers)} 篇")
            print(f"   • TLDR 生成: {tldr_count} 篇")
            print(f"   • 趋势分析维度: {analysis_count} 个")
            print("\n📁 报告位置:")
            for fmt, path in report_paths.items():
                print(f"   • [{fmt}] {path}")
            print("=" * 80 + "\n")

        except KeyboardInterrupt:
            logger.warning("\n用户中断程序执行")
            print("\n⚠️  程序已被用户中断")
        except Exception as e:
            logger.error(f"研究趋势分析出错: {e}", exc_info=True)
            print(f"\n❌ 研究趋势分析失败: {e}")
            import traceback

            traceback.print_exc()

            self._send_error_notification(str(e))
            raise

    # ==================== TLDR 生成辅助 ====================

    def _generate_tldrs_sequential(self, agent: TrendAgent, papers: list) -> Dict[str, str]:
        """顺序生成 TLDR"""
        tldrs = {}
        total = len(papers)
        with tqdm(total=total, desc="📝 生成 TLDR", unit="篇", ncols=100) as pbar:
            for idx, paper in enumerate(papers, 1):
                tldr = agent.generate_tldr(paper)
                if tldr:
                    tldrs[paper.paper_id] = tldr
                    logger.info(f"  [{idx}/{total}] {paper.title[:55]}...")
                pbar.update(1)
        return tldrs

    def _generate_tldrs_concurrent(self, agent: TrendAgent, papers: list) -> Dict[str, str]:
        """并发生成 TLDR"""
        tldrs = {}
        workers = min(self.settings.CONCURRENCY_WORKERS, self.settings.RESEARCH_TLDR_BATCH_SIZE)
        logger.info(f"  使用并发模式 (workers={workers})")

        with tqdm(total=len(papers), desc="📝 生成 TLDR", unit="篇", ncols=100) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(agent.generate_tldr, paper): paper for paper in papers}
                for future in as_completed(futures):
                    paper = futures[future]
                    try:
                        tldr = future.result()
                        if tldr:
                            tldrs[paper.paper_id] = tldr
                    except Exception as e:
                        logger.error(f"TLDR 生成异常 ({paper.title[:30]}...): {e}")
                    pbar.update(1)
        return tldrs

    # ==================== 相关性评分 + 截断 ====================

    def _get_source_limit(self, source_name: str) -> int:
        """
        获取指定来源的抓取上限：
        1) 优先使用 max_results_per_source[source_name]
        2) 回退到 pipeline 级 max_results
        """
        return int(self.max_results_per_source.get(source_name, self.max_results))

    def _fetch_candidate_papers(self, defer_history_mark: bool) -> Tuple[list, Dict[str, Any]]:
        """
        获取候选论文。

        兼容两种模式：
        - 默认 arXiv-only（保持原有 trend_research 行为）
        - topic discovery 多源模式：按 enabled_sources 抓取 arXiv + OpenAlex 期刊
        """
        enabled_set = set(self.enabled_sources or ["arxiv"])
        has_journal_in_enabled = any(s != "arxiv" and s in JOURNAL_ISSN_MAP for s in enabled_set)
        has_explicit_journals = any(j in JOURNAL_ISSN_MAP for j in (self.journals or []))
        use_multi_source = (
            len(enabled_set) > 1
            or "arxiv" not in enabled_set
            or has_journal_in_enabled
            or has_explicit_journals
        )

        if not use_multi_source:
            arxiv_source = ArxivSource(
                history_dir=self.history_dir,
                max_results=self._get_source_limit("arxiv"),
            )
            papers = arxiv_source.search_by_keywords(
                keywords=self.keywords,
                date_from=self.date_from,
                date_to=self.date_to,
                sort_order=self.sort_order,
                max_results=self._get_source_limit("arxiv"),
                categories=self.categories,
                use_history=self.dedupe_history,
                match_mode=self.match_mode,
                mark_after_fetch=not defer_history_mark,
            )
            return papers, {"arxiv": arxiv_source}

        logger.info(f"  多源模式启用: {sorted(enabled_set)}")
        days = max(1, (self.date_to - self.date_from).days + 1)
        papers: List[Any] = []
        handlers: Dict[str, Any] = {}

        # 统一去重键：优先 paper_id，避免跨源重复论文
        seen_ids = set()

        def _append_papers(items: List[Any]):
            for p in items:
                pid = getattr(p, "paper_id", "")
                if not pid or pid in seen_ids:
                    continue
                # 多源抓取统一收敛到用户指定时间窗
                if p.published_date:
                    if p.published_date.date() < self.date_from or p.published_date.date() > self.date_to:
                        continue
                seen_ids.add(pid)
                papers.append(p)

        if "arxiv" in enabled_set:
            arxiv_source = ArxivSource(
                history_dir=self.history_dir,
                max_results=self._get_source_limit("arxiv"),
            )
            handlers["arxiv"] = arxiv_source
            arxiv_papers = arxiv_source.search_by_keywords(
                keywords=self.keywords,
                date_from=self.date_from,
                date_to=self.date_to,
                sort_order=self.sort_order,
                max_results=self._get_source_limit("arxiv"),
                categories=self.categories,
                use_history=self.dedupe_history,
                match_mode=self.match_mode,
                mark_after_fetch=not defer_history_mark,
            )
            logger.info(f"  [arxiv] 候选 {len(arxiv_papers)} 篇")
            _append_papers(arxiv_papers)

        # 期刊来源可来自 enabled_sources 和 journals 两处，做并集
        journal_codes: List[str] = []
        for s in self.enabled_sources:
            if s != "arxiv" and s in JOURNAL_ISSN_MAP and s not in journal_codes:
                journal_codes.append(s)
        for j in self.journals:
            if j in JOURNAL_ISSN_MAP and j not in journal_codes:
                journal_codes.append(j)

        for journal in journal_codes:
            journal_source = OpenAlexSource(
                history_dir=self.history_dir,
                journals=[journal],
                max_results=self._get_source_limit(journal),
                email=self.openalex_email,
                api_key=self.openalex_api_key,
            )
            handlers[journal] = journal_source
            try:
                journal_papers = journal_source.fetch_papers(
                    days=days,
                    journals=[journal],
                    date_from=self.date_from,
                    date_to=self.date_to,
                    keywords=self.keywords,
                    match_mode=self.match_mode,
                )
                logger.info(
                    f"  [{journal}] 候选 {len(journal_papers)} 篇 (limit={self._get_source_limit(journal)})"
                )
                _append_papers(journal_papers)
            finally:
                try:
                    journal_source.close()
                except Exception:
                    pass

        papers.sort(
            key=lambda p: p.published_date or datetime.min,
            reverse=(self.sort_order == "descending"),
        )
        logger.info(f"  多源候选总数: {len(papers)} 篇")
        return papers, handlers

    def _mark_papers_history(self, papers: List[Any], handlers: Dict[str, Any]) -> None:
        """按 paper.source 写入对应历史。"""
        for p in papers:
            source_key = "arxiv" if p.source == "arxiv" else p.source
            handler = handlers.get(source_key)
            if handler:
                handler.mark_as_processed(p.paper_id)

    def _rerank_and_truncate(self, papers: list, top_n: int) -> list:
        """
        使用 AnalysisAgent 对候选论文按关键词加权评分并截断到 Top-N。

        评分结果通过本地 JSON 缓存（score_cache.json）持久化，按 paper_id 索引：
        - 命中缓存的论文跳过 LLM 调用；
        - 仅对新论文调用 cheap LLM 打分，打完合并写回缓存。
        关键词集合变更时（keywords_hash 不匹配）旧缓存整体失效。

        关键词权重取自 settings.PRIMARY_KEYWORD_WEIGHT（与 daily_research 打分口径一致）；
        评分失败的论文按 0 分处理，不影响整体排序。
        """
        from agents import AnalysisAgent

        try:
            weight = float(getattr(self.settings, "PRIMARY_KEYWORD_WEIGHT", 1.0) or 1.0)
        except Exception:
            weight = 1.0
        keywords_dict = {kw: weight for kw in self.keywords}

        # 加载评分缓存
        cache = _ScoreCache(
            path=self.history_dir / "score_cache.json",
            current_keywords_hash=_keywords_hash(self.keywords),
        )
        cache.load()

        # 拆分命中 / 未命中
        hits: List[tuple] = []
        misses: list = []
        for paper in papers:
            cached = cache.get(paper.paper_id)
            if cached is not None:
                hits.append((paper, cached))
            else:
                misses.append(paper)

        logger.info(
            f"  [cache] 评分缓存命中: {len(hits)}/{len(papers)}，"
            f"需要 LLM 打分: {len(misses)} 篇"
        )

        agent = AnalysisAgent() if misses else None
        # (paper, score_or_None)：score 为 None 表示本次 LLM 失败，不应写入缓存
        new_scored: List[tuple] = []

        def _score_one(paper):
            try:
                resp = agent.score_paper_with_keywords(
                    title=paper.title,
                    authors=paper.get_authors_string(),
                    abstract=paper.abstract or "",
                    keywords_dict=keywords_dict,
                )
                return paper, float(getattr(resp, "total_score", 0.0) or 0.0)
            except Exception as e:
                logger.warning(f"  相关性评分失败 ({paper.title[:30]}...): {e}")
                # 返回 None 表示失败；不会写入缓存，下次运行会重试
                return paper, None

        if misses:
            if self.settings.ENABLE_CONCURRENCY and len(misses) > 1:
                workers = min(self.settings.CONCURRENCY_WORKERS, len(misses))
                logger.info(f"  使用并发模式 (workers={workers})")
                with tqdm(total=len(misses), desc="🎯 相关性评分", unit="篇", ncols=100) as pbar:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(_score_one, p) for p in misses]
                        for future in as_completed(futures):
                            try:
                                new_scored.append(future.result())
                            except Exception as e:
                                logger.warning(f"  评分任务异常: {e}")
                            pbar.update(1)
            else:
                with tqdm(total=len(misses), desc="🎯 相关性评分", unit="篇", ncols=100) as pbar:
                    for paper in misses:
                        new_scored.append(_score_one(paper))
                        pbar.update(1)

            # 只把成功评分写入缓存并持久化；失败（score=None）跳过，下次重试
            success_count = 0
            fail_count = 0
            for paper, score in new_scored:
                if score is None:
                    fail_count += 1
                    continue
                cache.set(paper.paper_id, score)
                success_count += 1
            if fail_count:
                logger.warning(
                    f"  [cache] 本次评分失败 {fail_count} 篇，未写入缓存（下次会重试）"
                )
            if success_count:
                cache.save()

        # 合并缓存命中 + 本次新评分（失败的 None 视为 0 分参与本次排序，但不进缓存）
        merged: List[tuple] = list(hits)
        for paper, score in new_scored:
            merged.append((paper, score if score is not None else 0.0))
        merged.sort(key=lambda x: x[1], reverse=True)
        for paper, score in merged[:top_n]:
            logger.info(f"    ✓ 相关性 {score:.1f}  {paper.title[:60]}")

        # 保留本次重排序产生的分数，供通知阶段按 paper_id 查找
        self._last_scores = {p.paper_id: float(s) for p, s in merged[:top_n]}

        return [p for p, _ in merged[:top_n]]

    # ==================== 通知 ====================

    def _send_result_notification(
        self,
        total_papers: int,
        report_paths: Dict[str, Any],
        success: bool,
        trend_skills_count: int = 0,
        tldr_count: int = 0,
        token_usage: Dict[str, Any] = None,
        top_papers: Optional[List[Dict[str, Any]]] = None,
    ):
        """发送研究趋势分析结果通知"""
        if not self.settings.ENABLE_NOTIFICATIONS:
            return

        try:
            from notifications.notifier import _load_template, _render_template, TrendRunResult

            result = TrendRunResult(
                run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                keywords=self.keywords,
                date_from=str(self.date_from),
                date_to=str(self.date_to),
                total_papers=total_papers,
                tldr_count=tldr_count,
                trend_skills_count=trend_skills_count,
                report_paths={k: str(v) for k, v in report_paths.items()},
                success=success,
                token_usage=token_usage or {},
                top_papers=top_papers or [],
            )

            notifier = NotifierAgent()
            notifier.notify_trend(result)
            logger.info("通知发送完成")
        except Exception as e:
            logger.warning(f"通知发送失败: {e}")

    def _send_error_notification(self, error_msg: str):
        """发送错误通知"""
        if not self.settings.ENABLE_NOTIFICATIONS:
            return

        try:
            notifier = NotifierAgent()
            notifier.notify_error(
                "error_generic",
                error_type="研究趋势分析错误",
                error_message=error_msg,
                context=f"关键词: {', '.join(self.keywords)}, 时间: {self.date_from}~{self.date_to}",
            )
        except Exception:
            pass

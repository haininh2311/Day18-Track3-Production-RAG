"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load cross-encoder model, ưu tiên CrossEncoder của sentence-transformers vì ổn định hơn."""
        if self._model is None:
            try:
                # Option A: sentence_transformers CrossEncoder (ổn định hơn với các bản transformers mới)
                from sentence_transformers import CrossEncoder
                self._model = ("cross", CrossEncoder(self.model_name))
            except ImportError:
                try:
                    # Option B: FlagEmbedding
                    from FlagEmbedding import FlagReranker
                    self._model = ("flag", FlagReranker(self.model_name, use_fp16=True))
                except ImportError:
                    # Fallback: không có model nào — rerank bằng original_score
                    self._model = ("none", None)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        So sánh (query, document) theo từng cặp dùng cross-encoder,
        cho điểm chính xác hơn bi-encoder vì có thể xem xét tương tác giữa query và doc.
        """
        if not documents:
            return []

        model_type, model = self._load_model()

        pairs = [(query, doc["text"]) for doc in documents]

        if model_type == "flag" and model is not None:
            scores = model.compute_score(pairs)
            # FlagReranker có thể trả về 1 float nếu chỉ có 1 cặp
            if isinstance(scores, float):
                scores = [scores]
        elif model_type == "cross" and model is not None:
            scores = model.predict(pairs).tolist()
        else:
            # Fallback: giữ nguyên thứ tự gốc
            scores = [doc.get("score", 0.0) for doc in documents]

        # Gộp điểm với document gốc và sắp xếp
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored_docs[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank using flashrank — nhẹ, nhanh, phù hợp production."""
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()

            passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
            request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(request)

            reranked = []
            for rank, r in enumerate(results[:top_k]):
                orig_doc = documents[r["id"]]
                reranked.append(RerankResult(
                    text=orig_doc["text"],
                    original_score=float(orig_doc.get("score", 0.0)),
                    rerank_score=float(r.get("score", 0.0)),
                    metadata=orig_doc.get("metadata", {}),
                    rank=rank,
                ))
            return reranked
        except ImportError:
            # Fallback nếu flashrank chưa cài
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                )
                for i, doc in enumerate(documents[:top_k])
            ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs.

    Chạy n_runs lần và tính avg/min/max latency (ms).
    Lần đầu thường chậm hơn do load model — có thể bỏ qua lần đầu.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return {
        "avg_ms": mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_runs": n_runs,
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, docs)
    for r in results:
        print(f"[{r.rank}] score={r.rerank_score:.4f} | {r.text}")

    print("\nBenchmark:")
    stats = benchmark_reranker(reranker, query, docs, n_runs=3)
    print(f"  avg={stats['avg_ms']:.1f}ms  min={stats['min_ms']:.1f}ms  max={stats['max_ms']:.1f}ms")

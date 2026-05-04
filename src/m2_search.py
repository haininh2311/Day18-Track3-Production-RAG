"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words using underthesea.
    
    Returns a string containing both:
    1. Segmented tokens (joined by underscores, e.g., 'nghỉ_phép')
    2. Original tokens (split by spaces)
    This increases BM25 recall when segmentation is inconsistent.
    """
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        # Kết hợp cả text đã segment và text gốc để tăng recall
        return f"{segmented} {text}"
    except Exception:
        return text



class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks.

        Segments each chunk's text using Vietnamese word segmentation, then
        tokenizes by whitespace for BM25Okapi.
        """
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"].lower()).split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25 with Vietnamese-segmented query."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query.lower()).split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0  # chỉ lấy kết quả có điểm > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant with dense embeddings.

        Encodes all chunk texts using bge-m3 model, then uploads as
        PointStruct objects to a Qdrant collection with cosine distance.
        """
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Tạo mới (hoặc xóa và tạo lại) collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=True, batch_size=32)

        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]

        # Upload theo batch tránh timeout
        batch_size = 100
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection,
                points=points[start:start + batch_size],
            )

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vector similarity in Qdrant."""
        encoder = self._get_encoder()
        query_vector = encoder.encode(query).tolist()

        resp = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )

        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                method="dense",
            )
            for hit in resp.points
        ]



def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1).

    Đây là cách gộp kết quả từ nhiều hệ thống tìm kiếm khác nhau mà không
    cần chuẩn hóa điểm số. Công thức đơn giản nhưng rất hiệu quả trong thực tế.
    """
    # text → {"score": float, "result": SearchResult}
    rrf_scores: dict[str, dict] = {}

    for results in results_list:
        for rank, result in enumerate(results):
            key = result.text  # dùng text làm key để dedup
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            # Cộng dồn điểm RRF từ mỗi danh sách kết quả
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # Sắp xếp giảm dần theo điểm RRF tổng hợp
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    return [
        SearchResult(
            text=entry["result"].text,
            score=entry["score"],
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in sorted_results[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    text = "Nhân viên được nghỉ phép năm"
    print(f"Original:  {text}")
    print(f"Segmented: {segment_vietnamese(text)}")

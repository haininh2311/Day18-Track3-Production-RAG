"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # 1. Tách thành câu
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # Fallback nếu chỉ có 1 câu
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    # 2. Encode câu thành embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")  # model nhỏ, nhanh
        embeddings = model.encode(sentences, show_progress_bar=False)

        def cosine_sim(a, b) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        # 3. Gom câu vào nhóm theo ngưỡng similarity
        chunks: list[Chunk] = []
        current_group: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                # Bắt đầu chunk mới khi similarity thấp hơn ngưỡng
                chunks.append(Chunk(
                    text=" ".join(current_group),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                ))
                current_group = []
            current_group.append(sentences[i])

        # Đừng quên nhóm cuối cùng
        if current_group:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))

        return chunks

    except ImportError:
        # Fallback về chunk_basic nếu sentence_transformers chưa cài
        return chunk_basic(text, metadata=metadata)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    # 1. Gom paragraphs thành parent chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_parent_text = ""
    p_index = 0

    for para in paragraphs:
        # Nếu thêm paragraph này sẽ vượt ngưỡng → flush parent hiện tại
        if len(current_parent_text) + len(para) > parent_size and current_parent_text:
            pid = f"parent_{p_index}"
            parent = Chunk(
                text=current_parent_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                parent_id=pid,
            )
            parents.append(parent)

            # 2. Chia parent thành children bằng sliding window
            parent_text = current_parent_text.strip()
            for c_start in range(0, len(parent_text), child_size):
                child_text = parent_text[c_start:c_start + child_size].strip()
                if child_text:
                    children.append(Chunk(
                        text=child_text,
                        metadata={
                            **metadata,
                            "chunk_type": "child",
                            "parent_id": pid,
                            "child_index": len([c for c in children if c.parent_id == pid]),
                        },
                        parent_id=pid,
                    ))

            current_parent_text = ""
            p_index += 1

        current_parent_text += para + "\n\n"

    # Flush parent cuối cùng còn lại
    if current_parent_text.strip():
        pid = f"parent_{p_index}"
        parent = Chunk(
            text=current_parent_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            parent_id=pid,
        )
        parents.append(parent)

        parent_text = current_parent_text.strip()
        for c_start in range(0, len(parent_text), child_size):
            child_text = parent_text[c_start:c_start + child_size].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                        "parent_id": pid,
                        "child_index": len([c for c in children if c.parent_id == pid]),
                    },
                    parent_id=pid,
                ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # 1. Tách theo markdown headers (h1, h2, h3)
    # Giữ lại separator (headers) trong kết quả để biết section nào là gì
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Nếu đang có nội dung → flush thành chunk
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={
                        **metadata,
                        "section": current_header.strip(),
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Đừng quên section cuối cùng
    if current_content.strip() or current_header:
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={
                **metadata,
                "section": current_header.strip(),
                "strategy": "structure",
                "chunk_index": len(chunks),
            },
        ))

    # Nếu không có header nào, fallback về basic
    if not chunks:
        return chunk_basic(text, metadata=metadata)

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {}

    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_hierarchical_children: list[Chunk] = []
    all_structure: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})

        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        _, children = chunk_hierarchical(text, metadata=meta)
        all_hierarchical_children.extend(children)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    def get_stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_len": int(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    results["basic"] = get_stats(all_basic)
    results["semantic"] = get_stats(all_semantic)
    results["hierarchical"] = get_stats(all_hierarchical_children)
    results["structure"] = get_stats(all_structure)

    # In bảng so sánh
    print(f"\n{'Strategy':<15} {'Chunks':>7} {'Avg Len':>9} {'Min':>7} {'Max':>7}")
    print("-" * 48)
    for name, stats in results.items():
        print(f"{name:<15} {stats['num_chunks']:>7} {stats['avg_len']:>9} "
              f"{stats['min_len']:>7} {stats['max_len']:>7}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")

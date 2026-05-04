"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    """Helper tạo OpenAI client, raise ImportError nếu chưa cài."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Chỉ trả về tóm tắt, không giải thích thêm.",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback extractive: lấy 2 câu đầu
        sentences = text.split(". ")
        return ". ".join(sentences[:2]).strip() + ("." if len(sentences) > 1 else "")


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số, không giải thích thêm."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip()
        questions = [
            q.strip().lstrip("0123456789.-) ").strip()
            for q in raw.split("\n")
            if q.strip()
        ]
        return questions[:n_questions]
    except Exception:
        # Fallback: trả về danh sách rỗng
        return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    try:
        client = _get_openai_client()
        prompt_content = f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}" if document_title else f"Đoạn văn:\n{text}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Viết 1 câu ngắn (tối đa 20 từ) mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                        "Chỉ trả về 1 câu, không giải thích, không dấu ngoặc kép."
                    ),
                },
                {"role": "user", "content": prompt_content},
            ],
            max_tokens=80,
            temperature=0.3,
        )
        context = resp.choices[0].message.content.strip()
        return f"{context}\n\n{text}"
    except Exception:
        # Fallback: trả về text gốc không prepend
        return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ với các trường: '
                        '{"topic": "string", "entities": ["list", "of", "entities"], '
                        '"category": "policy|hr|it|finance|other", "language": "vi|en"}. '
                        'Chỉ trả về JSON, không giải thích thêm.'
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Xóa markdown code fence nếu có
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        # Fallback: trả về dict rỗng
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_single_chunk(chunk, idx, total, apply_all, methods):
    """Xử lý làm giàu cho 1 chunk (để chạy song song)."""
    text = chunk["text"]
    meta = chunk.get("metadata", {})
    doc_title = meta.get("source", "")

    # 1. Summary
    summary = ""
    if apply_all or "summary" in methods:
        summary = summarize_chunk(text)

    # 2. Hypothesis Questions (HyQA)
    questions: list[str] = []
    if apply_all or "hyqa" in methods:
        questions = generate_hypothesis_questions(text)

    # 3. Contextual Prepend
    enriched_text = text
    if apply_all or "contextual" in methods:
        enriched_text = contextual_prepend(text, document_title=doc_title)

    # 4. Auto Metadata
    auto_meta: dict = {}
    if apply_all or "metadata" in methods:
        auto_meta = extract_metadata(text)

    return EnrichedChunk(
        original_text=text,
        enriched_text=enriched_text,
        summary=summary,
        hypothesis_questions=questions,
        auto_metadata={**meta, **auto_meta},
        method="+".join(methods),
    )


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    max_workers: int = 10,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks (sử dụng đa luồng để tăng tốc).

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
        max_workers: Số luồng (threads) chạy song song.

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    apply_all = "full" in methods
    total = len(chunks)
    results = [None] * total

    print(f"  Starting multi-threaded enrichment ({max_workers} workers)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_chunk, chunk, i, total, apply_all, methods): i
            for i, chunk in enumerate(chunks)
        }

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"\n  [ERROR] Lỗi khi enrich chunk {idx+1}: {e}")
            
            completed += 1
            print(f"  Completed {completed}/{total} chunks", end="\r")

    print()  # newline sau progress bar
    return [r for r in results if r is not None]


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

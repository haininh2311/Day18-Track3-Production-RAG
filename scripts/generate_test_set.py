"""
Script: Generate test_set.json từ nội dung tài liệu thực.
Dùng OpenAI để sinh câu hỏi + ground_truth từ các đoạn văn trong data/.

Chạy: python scripts/generate_test_set.py
"""

import os
import sys
import glob
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def load_md_files(data_dir: str) -> list[dict]:
    """Load all .md files from data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({
                "text": f.read(),
                "source": os.path.basename(fp),
            })
    return docs


def split_into_sections(text: str, max_chars: int = 1500) -> list[str]:
    """Chia text thành các đoạn không quá max_chars để gửi lên API."""
    # Ưu tiên tách theo header markdown
    sections = re.split(r'(?=\n#{1,3} )', '\n' + text)
    result = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            result.append(section)
        else:
            # Tách thêm theo đoạn văn nếu section quá dài
            for i in range(0, len(section), max_chars):
                chunk = section[i:i + max_chars].strip()
                if chunk:
                    result.append(chunk)
    return result


def generate_qa_from_section(section: str, source: str, n_pairs: int = 2) -> list[dict]:
    """Dùng OpenAI để sinh câu hỏi + ground_truth từ 1 đoạn văn."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Dựa trên đoạn văn từ tài liệu '{source}', hãy tạo đúng {n_pairs} cặp câu hỏi-trả lời.\n"
                        "Yêu cầu:\n"
                        "- Câu hỏi phải cụ thể, có thể trả lời được từ đoạn văn\n"
                        "- Ground truth phải là câu trả lời ngắn gọn, chính xác, trích trực tiếp từ tài liệu\n"
                        "- Trả về JSON array: [{\"question\": \"...\", \"ground_truth\": \"...\"}]\n"
                        "- Chỉ trả về JSON, không giải thích thêm"
                    ),
                },
                {"role": "user", "content": f"Đoạn văn:\n{section}"},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        raw = resp.choices[0].message.content.strip()
        # Xóa markdown code fence nếu có
        raw = raw.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(raw)

        # Validate format
        valid = []
        for p in pairs:
            if isinstance(p, dict) and "question" in p and "ground_truth" in p:
                if p["question"].strip() and p["ground_truth"].strip():
                    valid.append({
                        "question": p["question"].strip(),
                        "ground_truth": p["ground_truth"].strip(),
                        "source": source,
                    })
        return valid

    except Exception as e:
        print(f"    ⚠️  Error generating QA: {e}")
        return []


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_set.json")

    docs = load_md_files(data_dir)
    if not docs:
        print("❌ Không tìm thấy file .md trong data/")
        print("   Hãy chạy: python scripts/convert_pdfs.py trước")
        return

    print(f"Found {len(docs)} document(s). Generating test set...\n")

    all_pairs: list[dict] = []
    target_per_doc = max(10, 20 // len(docs))  # Ít nhất 20 câu tổng cộng

    for doc in docs:
        if len(doc["text"]) < 200:
            print(f"   [SKIP] {doc['source']} (too small/empty)")
            continue
            
        print(f"   Processing: {doc['source']}")
        sections = split_into_sections(doc["text"])
        print(f"   {len(sections)} sections found")

        # Chọn đều các section để cover toàn bộ tài liệu
        step = max(1, len(sections) // target_per_doc)
        selected = sections[::step][:target_per_doc]

        doc_pairs = []
        for i, section in enumerate(selected):
            print(f"   Generating QA for section {i+1}/{len(selected)}...", end="\r")
            pairs = generate_qa_from_section(section, doc["source"], n_pairs=2)
            doc_pairs.extend(pairs)

        print(f"\n   [OK] Generated {len(doc_pairs)} QA pairs from {doc['source']}")
        all_pairs.extend(doc_pairs)

    # Loại bỏ duplicate và giới hạn tối đa 30 câu
    seen_questions = set()
    unique_pairs = []
    for p in all_pairs:
        q = p["question"].lower().strip()
        if q not in seen_questions:
            seen_questions.add(q)
            unique_pairs.append({"question": p["question"], "ground_truth": p["ground_truth"]})

    unique_pairs = unique_pairs[:30]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved {len(unique_pairs)} QA pairs to: {output_path}")
    print("\nSample questions:")
    for p in unique_pairs[:3]:
        print(f"  Q: {p['question']}")
        print(f"  A: {p['ground_truth'][:80]}...")
        print()


if __name__ == "__main__":
    main()

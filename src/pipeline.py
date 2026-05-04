"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1) - Rất nhanh, cần để lấy chunks cho BM25
    print("\n[1/3] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5) - Tốn phí OpenAI, có thể comment nếu đã chạy rồi
    # print("\n[2/4] Enriching chunks (M5)... [ĐÃ BẬT ĐA LUỒNG]")
    # enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    # if enriched:
    #     all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
    #     print(f"  Enriched {len(enriched)} chunks")

    # Step 3: Index (M2)
    print("\n[3/4] Initializing Search...")
    search = HybridSearch()
    # Nạp lại BM25 (vì BM25 lưu trong RAM, sẽ mất khi tắt script)
    print("  Indexing BM25 (In-memory)...")
    search.bm25.index(all_chunks)
    
    # Bỏ qua việc index vào Qdrant nếu bạn chắc chắn dữ liệu đã có sẵn trong Qdrant
    # print("  Indexing Dense (Qdrant)...")
    # search.dense.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker



def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # LLM generation — dùng OpenAI gpt-4o-mini để sinh câu trả lời từ context
    # Điều này giúp tăng điểm Faithfulness và Answer Relevancy đáng kể
    try:
        from openai import OpenAI
        from config import OPENAI_API_KEY
        client = OpenAI(api_key=OPENAI_API_KEY)
        context_str = "\n\n---\n\n".join(contexts)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu nội bộ. "
                        "Quy tắc quan trọng:\n"
                        "1. CHỈ trả lời dựa trên thông tin có trong CONTEXT bên dưới.\n"
                        "2. Nếu context không có thông tin → trả lời: 'Không tìm thấy thông tin liên quan trong tài liệu.'\n"
                        "3. Câu trả lời ngắn gọn, súc tích, đúng trọng tâm.\n"
                        "4. Không suy đoán hoặc thêm thông tin ngoài context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_str}\n\nCÂU HỎI: {query}",
                },
            ],
            temperature=0.1,  # thấp để hạn chế hallucination
            max_tokens=512,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        # Fallback: dùng context đầu tiên nếu OpenAI lỗi
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts



def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        status = "[OK]" if s >= 0.75 else "[FAIL]"
        print(f"  {status} {m}: {s:.4f}")


    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")

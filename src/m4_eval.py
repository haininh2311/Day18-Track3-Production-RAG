"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation (4 metrics)."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from config import OPENAI_API_KEY

        # Explicitly setup LLM and Embeddings for RAGAS
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
        
        # Thêm per_question results để analyze
        df = result.to_pandas()
        
        per_question: list[EvalResult] = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            ))

        return {
            "faithfulness": float(df["faithfulness"].mean() if "faithfulness" in df else 0.0),
            "answer_relevancy": float(df["answer_relevancy"].mean() if "answer_relevancy" in df else 0.0),
            "context_precision": float(df["context_precision"].mean() if "context_precision" in df else 0.0),
            "context_recall": float(df["context_recall"].mean() if "context_recall" in df else 0.0),
            "per_question": per_question,
        }


    except ImportError as e:
        print(f"  [WARN] RAGAS/datasets not installed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }
    except Exception as e:
        print(f"  [ERROR] RAGAS evaluation error: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }



def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Với mỗi câu hỏi thất bại, xác định metric nào tệ nhất và đưa ra
    chẩn đoán + gợi ý sửa đổi theo Diagnostic Tree.
    """
    if not eval_results:
        return []

    # Diagnostic Tree: map metric → (diagnosis, suggested_fix)
    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating — câu trả lời không dựa trên context",
            "Tighten prompt, lower temperature, thêm instruction 'Chỉ trả lời dựa trên context'",
        ),
        "context_recall": (
            "Missing relevant chunks — retrieval bỏ sót thông tin quan trọng",
            "Improve chunking strategy, tăng top_k, thêm BM25 hoặc query expansion",
        ),
        "context_precision": (
            "Too many irrelevant chunks — retrieval lấy quá nhiều noise",
            "Add reranking, giảm top_k, thêm metadata filter hoặc hybrid search",
        ),
        "answer_relevancy": (
            "Answer doesn't match question — câu trả lời lạc đề",
            "Improve prompt template, thêm explicit instruction về format câu trả lời",
        ),
    }

    # Ngưỡng để chẩn đoán lỗi
    thresholds = {
        "faithfulness": 0.85,
        "context_recall": 0.75,
        "context_precision": 0.75,
        "answer_relevancy": 0.80,
    }

    # 1. Tính avg_score cho mỗi câu và sắp xếp tăng dần
    scored = []
    for r in eval_results:
        avg = mean([r.faithfulness, r.answer_relevancy, r.context_precision, r.context_recall])
        scored.append((avg, r))

    scored.sort(key=lambda x: x[0])  # ascending → câu tệ nhất đầu tiên
    bottom = scored[:bottom_n]

    # 2. Phân tích từng câu thất bại
    failures = []
    for avg_score, r in bottom:
        metric_scores = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy,
        }

        # Tìm metric nào thấp nhất
        worst_metric = min(metric_scores, key=lambda m: metric_scores[m])
        worst_score = metric_scores[worst_metric]

        # Ưu tiên metric nào vi phạm ngưỡng (dưới threshold)
        violated = {
            m: s for m, s in metric_scores.items() if s < thresholds[m]
        }
        if violated:
            worst_metric = min(violated, key=lambda m: violated[m])
            worst_score = violated[worst_metric]

        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric,
            ("Unknown issue", "Review pipeline manually"),
        )

        failures.append({
            "question": r.question,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "all_scores": {m: round(s, 4) for m, s in metric_scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

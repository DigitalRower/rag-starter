import json
from pathlib import Path
from typing import cast

from evals.scorer import score_answer_relevance, score_faithfulness, score_precision
from rag_starter import query


def load_dataset(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        dataset = cast(list[dict[str, str]], json.load(f))
    return dataset

# returns ("faithfullness": 0.87, "answer_relevance": 0.82, ...)
def run_eval(
    dataset: list[dict[str, str]],
    collection: query.Collection
    ) -> list[dict]:

    results = []
    
    #for item in dataset[26:27]:    # use for test to save on tokens
    for item in dataset:
        # call query.main() to get answer + chunks
        generated_result = query.main(collection, item["question"])

        # extract text from chunks
        chunks_text = [c["text"] for c in generated_result["chunks"]]

        # call scorer.score_faithfulness() with question, chunks_text, answer, expected_answer
        faithfulness = score_faithfulness(item["question"], chunks_text, generated_result["answer"], item["expected_answer"])

        relevance = score_answer_relevance(item["question"], generated_result["answer"], item["expected_answer"])

        precision = score_precision(item["question"], chunks_text, item["expected_answer"])

        # collect the result
        results.append({
            # fields carried over from dataset
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            # fields added by the eval run
            "actual_answer": generated_result["answer"],
            "faithfulness_score": faithfulness["score"],
            "faithfulness_reasoning": faithfulness["reasoning"],
            "relevance_score": relevance["score"],
            "relevance_reasoning": relevance["reasoning"],
            "precision_score": precision["score"],
            "precision_reasoning": precision["reasoning"],
            "sources": generated_result["sources"]
        })     

    # return the collected results
    return results


def write_results(results: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    # make directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # write JSON (indent=2 keeps it human-readable)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def print_summary(results: list[dict]) -> None:
    categories = ["happy_path", "edge_case", "adversarial", "bias_paired"]
    print("\nCategory       Avg Faithfulness    Avg Relevance    Precision@3    Count")
    print("-" * 75)
    all_faith = []
    all_relev = []
    all_prec = []

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        count = len(cat_results)
        if count == 0:
            continue
        faith_scores = [r["faithfulness_score"] for r in cat_results]
        relev_scores = [r["relevance_score"] for r in cat_results]
        prec_scores = [r["precision_score"] for r in cat_results] 
        faith_avg = sum(faith_scores) / count
        relev_avg = sum(relev_scores) / count
        prec_avg = sum(prec_scores) / len(prec_scores) if prec_scores else None
        all_faith.extend(faith_scores)
        all_relev.extend(relev_scores)
        all_prec.extend(prec_scores)
        prec_display = f"{prec_avg:.2f}" if prec_avg is not None else "N/A"
        print(f"{cat:<20}{faith_avg:<20.2f}{relev_avg:<20.2f}{prec_display:<15}{count}")

    print("-" * 75)
    print(f"{'OVERALL':<20}{sum(all_faith)/len(all_faith):<20.2f}{sum(all_relev)/len(all_relev):<20.2f}{sum(all_prec)/len(all_prec):<15.2f}{len(all_faith)}")

def write_summary(results: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = ["happy_path", "edge_case", "adversarial", "bias_paired"]
    summary: dict[str, object] = {}
    all_faith, all_relev, all_prec = [], [], []
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        faith_scores = [r["faithfulness_score"] for r in cat_results]
        relev_scores = [r["relevance_score"] for r in cat_results]
        prec_scores = [r["precision_score"] for r in cat_results]
        all_faith.extend(faith_scores)
        all_relev.extend(relev_scores)
        all_prec.extend(prec_scores)
        summary[cat] = {
            "faithfulness": round(sum(faith_scores) / len(faith_scores), 2),
            "relevance": round(sum(relev_scores) / len(relev_scores), 2),
            "precision": round(sum(prec_scores) / len(prec_scores), 2),
            "count": len(cat_results),
        }
    summary["overall"] = {
        "faithfulness": round(sum(all_faith) / len(all_faith), 2),
        "relevance": round(sum(all_relev) / len(all_relev), 2),
        "precision": round(sum(all_prec) / len(all_prec), 2),
        "count": len(all_faith),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    DATASET_PATH = Path(__file__).parent / "dataset.json"
    dataset = load_dataset(DATASET_PATH)
    collection = query.get_collection()
    graded = run_eval(dataset, collection)
    RESULTS_PATH = Path(__file__).parent / "results" / "results.json"
    SUMMARY_PATH = Path(__file__).parent / "results" / "summary.json"
    write_results(graded, RESULTS_PATH)
    write_summary(graded, SUMMARY_PATH)
    print_summary(graded)
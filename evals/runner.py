import json

from rag_starter.query import main, get_collection

collection = get_collection()
result = main(collection, question)


def load_dataset(path: str) -> list[dict[str, str]]:
    pass

def run_eval(dataset: list[dict[str, str]]) -> dict[str, float]:
    # returns ("faiithfullness": 0.87, "answer_relevance": 0.82, ...)
    pass

def write_results(results: dict[str, float], output_path: str) -> None:
    pass

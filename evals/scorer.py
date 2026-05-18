import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()   

def score_faithfulness(
        question: str,
        retrieved_chunks: list[str], 
        generated_answer: str,
        expected_answer: str,
) -> dict[str, int | str]:
    context = "\n\n".join(retrieved_chunks)

    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=0, # deterministic scoring
        system=(
            "You are an evaluator. Given the following context, question, and answer, "
            "score the answer for faithfulness on a scale of 1–5. "
            "Faithfulness = the answer only uses information present in the provided context. "
            "Score 5 if fully grounded, 1 if it contains hallucinated facts not in context. "
            "Return JSON only: {\"score\": int, \"reasoning\": str}"
        ),
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {generated_answer}",
            }
        ],
    )

    raw = response.content[0].text
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


if __name__ == "__main__":
    # PYTHONPATH=src python evals/scorer.py
    chunks = ["Agents use skills to perform tasks. A skill is a reusable capability."]

    cases = [
        {
            "label": "faithful",
            "question": "What are agent skills?",
            "answer": "Agent skills are reusable capabilities that agents use to perform tasks.",
            "expected": "Agent skills are reusable capabilities.",
        },
        {
            "label": "unfaithful",
            "question": "What are agent skills?",
            "answer": "Agent skills include neural interfaces and quantum reasoning modules.",  # hallucinated
            "expected": "Agent skills are reusable capabilities.",
        },
        {
            "label": "i_dont_know",
            "question": "What are agent skills?",
            "answer": "I don't have enough information to answer that question.",
            "expected": "Agent skills are reusable capabilities.",
        },
    ]

    for case in cases:
        result = score_faithfulness(case["question"], chunks, case["answer"], case["expected"])
        print(f"[{case['label']}] score={result['score']} | {result['reasoning']}")



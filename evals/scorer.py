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


















# def build_scorer_prompt(question, context, answer, expected_answer):
#     user_content = f"""
#     System role: "You are an expert evaluator of AI-generated responses."

#     Task: "Given a question, the retrieved context, and a generated answer, 
#         score the answer for faithfulness."

#     Definition: "Faithfulness means the answer contains only information 
#                 present in the provided context. Hallucination (facts not 
#                 in context) results in lower scores."

#     Scale with rubric:
#     5 = Completely faithful; every fact is grounded in context
#     4 = Mostly faithful; one minor unsupported detail
#     3 = Partially faithful; 50/50 grounded vs hallucinated
#     2 = Mostly hallucinated; one or two facts from context
#     1 = Completely hallucinated; contradicts or ignores context

#     Input:
#     Question: [user's question]
#     Retrieved context: [the chunks from retrieval]
#     Generated answer: [the answer the system produced]

#     Output format: 
#     {
#         "score": <1-5>,
#         "reasoning": "<explain why you gave this score>"
#     } 

#     RUBRIC:
#     5 = Every fact in the answer is grounded in the provided context
#     4 = 1 minor detail is unsupported, but main points are grounded  
#     3 = About 50% grounded, 50% appears to be hallucinated
#     2 = 1-2 facts from context, but mostly hallucinated
#     1 = Completely hallucinated or contradicts context

#     Question: {question}

#     Retrieved context:
#     {context}

#     Generated answer:
#     {answer}

#     Respond ONLY with JSON:
#     {{"score": <int 1-5>, "reasoning": "<brief explanation>"}}
#     """
#     return user_content


   # - [ ] Build scorer.py — LLM-as-judge scoring for faithfulness:
#       - Takes: question, retrieved chunks, generated answer, expected answer
#       - Calls Claude (Haiku) with a grading rubric prompt
#       - Returns a score (1–5 or 0/1) + reasoning
#       - System prompt structure: "You are an evaluator. Given the following
#         context, question, and answer, score the answer for faithfulness on
#         a scale of 1–5. Faithfulness = the answer only uses information
#         present in the provided context. Score 5 if fully grounded, 1 if
#         it contains hallucinated facts not in context. Return JSON:
#         {score: int, reasoning: str}"




# def scorer(question, context, answer, expected_answer):
#     pass

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         user_question = " ".join(sys.argv[1:])

#         collection = get_collection()
#         result = main(collection, user_question)

#         print("\nAnswer:", result["answer"])
#         print("\nSources:", result["sources"])

#     else:
#         print("Error: No string provided.")





# - [ ] Test scorer.py on 3 hand-picked Q/A pairs before wiring to runner:
#       - 1 clearly faithful answer (expect score 4–5)
#       - 1 clearly unfaithful answer (manually inject hallucination — expect 1–2)
#       - 1 "I don't know" response (expect score 5 — correct behavior)

import sys
from typing import TypedDict

import chromadb
from anthropic import Anthropic, APIError
from anthropic.types import TextBlock
from chromadb import Collection
from dotenv import load_dotenv

load_dotenv()


class QueryResponse(TypedDict):
    answer: str
    sources: list[str]
    chunks: list[dict[str, str]]


def get_collection() -> Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(name="anthropic_docs")


def retrieve_chunks(collection: Collection, q: str, n_results: int = 3) -> list[dict[str, str]]:

    results = collection.query(query_texts=[q], n_results=n_results)

    docs = results["documents"]
    metas = results["metadatas"]

    if docs is None or metas is None:
        return []

    chunks = []
    for doc, meta in zip(docs[0], metas[0]):
        chunks.append({"text": doc, "source": str(meta.get("source", "unknown"))})

    return chunks


def build_prompt(user_question: str, chunks: list[dict[str, str]]) -> str:
    system_prompt = "You are a helpful assistant answering questions about Anthropic's documentation. Answer ONLY from the provided context. If the context doesn't contain the answer, say 'I don't know based on the provided documentation.'"

    context = "Context:\n"
    for i, item in enumerate(chunks):
        context += f" [{i}] (from {item['source']}): {item['text']}"

    prompt = system_prompt + "\n\n" + context + "\n\nUser question: " + user_question

    return prompt


def generate_answer(prompt: str) -> str:
    client = Anthropic()
    # COST NOTE: Anthropic prompt caching (cache_control breakpoints) will be
    # applied to this system prompt + retrieved context at W13 once we have
    # a measurable token baseline. At W4 scale, caching adds complexity without
    # a benchmark to justify it. See W13 hardening sprint.
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        block = message.content[0]
        return block.text if isinstance(block, TextBlock) else ""
    except APIError as e:
        return f"Couldn't reach Claude: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


def main(collection: Collection, user_question: str) -> QueryResponse:
    chunks = retrieve_chunks(collection, user_question)
    prompt = build_prompt(user_question, chunks)
    answer = generate_answer(prompt)

    sources = list(dict.fromkeys([item["source"] for item in chunks]))
    response: QueryResponse = {"answer": answer, "sources": sources, "chunks": chunks}

    return response


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])

        collection = get_collection()
        result = main(collection, user_question)

        print("\nAnswer:", result["answer"])
        print("\nSources:", result["sources"])

    else:
        print("Error: No string provided.")

import sys
import chromadb
from anthropic import Anthropic
from anthropic import APIError
from dotenv import load_dotenv

load_dotenv()   

def retrieve_chunks(collection, q, n_results=5):

    results = collection.query(
        query_texts=[q],
        n_results=n_results
    )
   
    docs = results['documents'][0]
    metas = results['metadatas'][0]

    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append({"text": doc, "source": meta.get("source", "unknown")})

    return(chunks)

def build_prompt(user_question, chunks):
    system_prompt = "You are a helpful assistant answering questions about Anthropic's documentation. Answer ONLY from the provided context. If the context doesn't contain the answer, say 'I don't know based on the provided documentation.'"

    context = "Context:\n"
    for i, item in enumerate(chunks):
        context += f" [{i}] (from {item['source']}): {item['text']}"
    
    prompt = system_prompt + "\n\n" + context + "\n\nUser question: " + user_question

    return prompt

def generate_answer(prompt):
    client = Anthropic()          
    ########## 1. Add a comment in src/query.py where you build the prompt, above the client.messages.create() or stream() call:
    # COST NOTE: Anthropic prompt caching (cache_control breakpoints) will be
    # applied to this system prompt + retrieved context at W13 once we have
    # a measurable token baseline. At W4 scale, caching adds complexity without
    # a benchmark to justify it. See W13 hardening sprint.
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": prompt 
                }
            ]
        )
        return(message.content[0].text)
    except APIError as e:
        return(f"Couldn't reach Claude: {e}")
    except Exception as e:
        return(f"Unexpected error: {e}")

def main(collection, user_question):
    chunks = retrieve_chunks(collection, user_question) 
    prompt = build_prompt(user_question, chunks)
    answer = generate_answer(prompt) 

    sources = [item['source'] for item in chunks]
    sources = list(dict.fromkeys([item['source'] for item in chunks]))
    response = {"answer": answer, "sources": sources}

    return response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
       
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="anthropic_docs")

        result = main(collection, user_question)
    
        print("\nAnswer:", result["answer"])
        print("\nSources:", result["sources"])

    else:
        print("Error: No string provided.")




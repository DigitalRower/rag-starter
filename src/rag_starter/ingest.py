from pathlib import Path

import chromadb


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[dict[str, str]]:
    """Split text into overlapping chunks."""
    chunks = []
    step_size = chunk_size - overlap

    for start_pos in range(0, len(text), step_size):
        chunk_text = text[start_pos : start_pos + chunk_size]
        chunk_dict = {"text": chunk_text}
        chunks.append(chunk_dict)

    return chunks


def ingest_documents() -> None:
    """Load markdown files, chunk them, and store in Chroma."""

    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./chroma_db/")
    collection = client.get_or_create_collection(name="anthropic_docs")

    # Load all markdown files
    data_dir = Path("./data")
    documents = []

    for file_path in data_dir.glob("*.md"):
        text = file_path.read_text()
        filename = file_path.name
        documents.append({"text": text, "source": filename})

    # Processs each document
    chunk_id = 0

    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        # Chunk the document
        chunks = chunk_text(text, chunk_size=1500, overlap=200)

        # Add each chunk to Chroma
        for chunk in chunks:
            chunk_id += 1
            chunk_text_content = chunk["text"]
            chunk_metadata = {"source": source}

            # Chroma's add() method stores the chunk
            collection.add(
                ids=[str(chunk_id)],
                documents=[chunk_text_content],
                metadatas=[chunk_metadata],
            )

            print(f"Added chunk {chunk_id} from {source}")

    print(f"Total chunks added {chunk_id}")

    # Verify
    count = collection.count()
    print(f"Total chunks in collection: {count}")

    # Test query
    query = "what can you tell me about the claude models?"
    test_results = collection.query(query_texts=[query], n_results=3)
    print(f"\nquery: {query}")
    result_docs = test_results["documents"]
    if result_docs is not None:
        print(f"\nTest query returned {len(result_docs[0])} results")
        for i, result_doc in enumerate(result_docs[0]):
            print(f"Result {i + 1}: {result_doc[:200]}...")


if __name__ == "__main__":
    ingest_documents()

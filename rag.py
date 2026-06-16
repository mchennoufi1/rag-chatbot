import os
import time
from dotenv import load_dotenv
import anthropic
import voyageai
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def get_embedding(text: str) -> list[float]:
    result = voyage.embed([text], model="voyage-3", input_type="document")
    return result.embeddings[0]


def process_and_store_pdf(filepath: str):
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.page_content)
        vectors.append({
            "id": f"{os.path.basename(filepath)}-chunk-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": filepath
            }
        })

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])

    return len(chunks)


def ask_question(query: str, history: list = []) -> dict:
    query_embedding = voyage.embed([query], model="voyage-3", input_type="query").embeddings[0]

    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
    sources = list(set(match["metadata"]["source"] for match in results["matches"]))
    context = "\n\n".join(context_chunks)

    # Build message history for Claude
    messages = []
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Answer the question based only on the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}"""
    })

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a helpful support assistant. Answer questions based on the provided document context. Be concise and accurate.",
        messages=messages
    )

    return {
        "answer": message.content[0].text,
        "sources": sources
    }
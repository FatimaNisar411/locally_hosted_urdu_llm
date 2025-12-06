import chromadb
import requests
from collections import Counter
import time
def get_chroma():
    # Use the existing collection in db folder
    # No need to specify embedding function when getting - it's stored with the collection
    client = chromadb.PersistentClient(path="db")
    collection = client.get_collection(name="tashreeh_chroma_bgem3")
    return collection




collection = get_chroma()
# ------------------------------------------------
# 1. Detect poem title (AUTO-EMBEDDING by Chroma)
# ------------------------------------------------
def detect_poem_title(user_couplet, collection):
    """
    Run similarity search using Chroma's internal embedder.
    Pick the top-1 most similar chunk.
    """
    results = collection.query(
        query_texts=[user_couplet],   # <--- AUTO-EMBEDDING happens here
        n_results=1
    )

    top_metadata = results["metadatas"][0][0]
    poem_title = top_metadata["poem"]

    print(f"📌 Detected poem: {poem_title}")
    return poem_title


# ------------------------------------------------
# 2. Retrieve ALL chunks of that poem
# ------------------------------------------------
def get_poem_chunks(poem_title, collection):
    """
    Pull all chunks where poem_title == requested poem.
    """
    results = collection.get(where={"poem": poem_title})
    


    docs = results["documents"]
    metas = results["metadatas"]

    # Sort by chunk_id if available
    combined = list(zip(docs, metas))
    combined_sorted = sorted(
        combined,
        key=lambda x: int(x[1].get("chunk_id", 0))
    )

    sorted_docs = [c[0] for c in combined_sorted]
    return sorted_docs


# ------------------------------------------------
# 3. Full retrieval pipeline (NO Alif)
# ------------------------------------------------
def retrieve_poem(user_couplet, collection):
    """
    Detect poem → fetch poem chunks → return them.
    """
    poem_title = detect_poem_title(user_couplet, collection)
    chunks = get_poem_chunks(poem_title, collection)

    print(f"📚 Retrieved {len(chunks)} chunks.")
    return {
        "title": poem_title,
        "chunks": chunks
    }
def rag_query(query, collection):
    

    print(f"\n🔍 Query: {query}")

    # ------------------------------------------------
    # 1. Detect poem using your function
    # ------------------------------------------------
    start_time = time.time()
    poem_title = detect_poem_title(query, collection)
    detection_time = time.time() - start_time
    print(f"⏱️ Detection took: {detection_time:.2f}s")

    # ------------------------------------------------
    # 2. Retrieve all chunks using your function
    # ------------------------------------------------
    chunks = get_poem_chunks(poem_title, collection)
    print(f"📚 Retrieved {len(chunks)} chunks for poem '{poem_title}'")

    # Join chunks into one context string
    context = "\n\n".join(chunks)
    print(f"📝 Total context length: {len(context)} characters")

    # ------------------------------------------------
    # 3. Build LLM prompt (improved)
    # ------------------------------------------------
    prompt = f"""
You are an expert on Bang-e-Dara by Allama Iqbal.

The context contains the FULL tashreeh of the poem, but the user is asking about ONE specific couplet.
You must extract ONLY the explanation for the user's couplet.

Context (تشریح):
{context}

User's Couplet:
{query}

Instructions:
- Focus ONLY on the user's couplet
- Use only the relevant part from the provided context
- Do NOT explain the whole nazm
- Keep it clear, correct, and simple
- Include meaning, message, and any philosophical point if mentioned in context
- No repetition

Answer in clear Urdu:
"""

    # ------------------------------------------------
    # 4. Send to local Llama
    # ------------------------------------------------
    print("\n🤖 Sending to Llama 3.1:8b...")
    start_time = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False}
    )
    llm_time = time.time() - start_time

    print(f"⏱️ LLM response took: {llm_time:.2f}s")
    print(f"✅ Total time: {detection_time + llm_time:.2f}s\n")

    return response.json()["response"]

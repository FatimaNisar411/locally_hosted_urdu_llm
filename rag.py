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
    # === Prompt ===
    system_prompt = """
آپ اردو ادب اور بالِ جبرئیل / بانگِ درا کے ماہر ہیں۔
دیے گئے کونٹیکسٹ (تشریح) کی مدد سے صارف کے سوال کا بہترین ممکنہ جواب دیں۔

— ہمیشہ صاف، خوبصورت اردو میں لکھیں
— صرف فراہم کردہ تشریح سے جواب بنائیں
— بے وجہ لمبی تکرار نہ کریں
— مفہوم، پیغام، اور فلسفہ واضح کریں
"""

    user_prompt = f"""
سوال:
{query}

تشریح (Context):
{context}

براہ کرم اس کی وضاحت اردو میں کریں:
"""

    # === Send to ALIF via LM Studio ===
    print("\n🤖 Sending to Alif (LM Studio)...")

    start_time = time.time()
    
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",  # or use your LAN IP
        json={
            "model": "alif-1.0-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
    ).json()

    llm_time = time.time() - start_time
    print(f"⏱️  LLM took: {llm_time:.2f}s")
    
    answer = response['choices'][0]['message']['content']
    print(f"\n✅ Response: {answer[:200]}...\n")
    
    return answer

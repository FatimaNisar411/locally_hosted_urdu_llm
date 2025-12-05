import chromadb
import requests


def get_chroma():
    # Use the existing collection in db folder
    # No need to specify embedding function when getting - it's stored with the collection
    client = chromadb.PersistentClient(path="db")
    collection = client.get_collection(name="tashreeh_chroma_bgem3")
    return collection




collection = get_chroma()

def rag_query(query):
    import time
    
    print(f"\n🔍 Query: {query}")
    
    # Retrieve
    start_time = time.time()
    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    retrieval_time = time.time() - start_time
    print(f"⏱️  Retrieval took: {retrieval_time:.2f}s")
    
    # Show retrieved chunks with distance scores
    print(f"\n📦 Retrieved {len(results['documents'][0])} chunks:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        preview = doc[:200].replace('\n', ' ')  # First 200 chars
        print(f"   {i}. [Distance: {distance:.3f}] {preview}...")
    
    contexts = "\n\n".join(results["documents"][0])
    print(f"\n📝 Total context length: {len(contexts)} characters")

    # Prompt
    prompt = f"""
You are an expert on Bang-e-Dara by Allama Iqbal. Use the provided Urdu commentary (تشریح) to explain the given verse or poem.

Context (تشریح):
{contexts}

User's Query:
{query}

Instructions:
- Extract the relevant explanation from the provided context
- Present it in clear, simple Urdu
- Focus on the meaning (مفہوم), message (پیغام), and significance (اہمیت)
- If the context mentions Iqbal's philosophy, include that
- Keep your response concise and accurate
- DO NOT repeat the same phrase multiple times
- Use ONLY the information from the context provided

Respond in clear Urdu:
"""

    # Local LLM request
    print(f"\n🤖 Sending to Llama 3.1:8b...")
    start_time = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False}
    )
    llm_time = time.time() - start_time
    print(f"⏱️  LLM response took: {llm_time:.2f}s")
    print(f"✅ Total time: {retrieval_time + llm_time:.2f}s\n")

    return response.json()["response"]

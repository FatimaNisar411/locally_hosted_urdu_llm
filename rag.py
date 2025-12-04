import chromadb
import requests
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


def get_chroma():
    # Use BGE-M3 embedding model via Ollama
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="bge-m3"
    )
    
    # Use the new PersistentClient API
    client = chromadb.Persist
    entClient(path="dba")
    
    # Delete old collection if it exists with wrong embedding function
    try:
        client.delete_collection(name="tashreeh_chroma_db_better")
    except:
        pass
    
    collection = client.get_or_create_collection(
        name="tashreeh_chroma_db_better",
        embedding_function=ollama_ef
    )
    return collection




collection = get_chroma()

def rag_query(query):
    # Retrieve
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    contexts = "\n\n".join(results["documents"][0])

    # Prompt
    prompt = f"""
آپ کا کام دی گئی تشریح کو آسان، سادہ اور عام فہم اردو میں دوبارہ بیان کرنا ہے۔

Context:
{contexts}

سوال:
{query}

آسان اور وضاحت کے ساتھ جواب دیں:
"""

    # Local LLM request
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False}
    )

    return response.json()["response"]

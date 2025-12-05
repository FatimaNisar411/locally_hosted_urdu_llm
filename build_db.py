"""
Build ChromaDB collection with BGE-M3 embeddings for Bang-e-Dara Tashreeh
"""
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_chromadb(text_file_path="tashreeh_extracted.txt", db_path="db"):
    """
    Save extracted poem + explanation text to ChromaDB using BGE-M3 embeddings via SentenceTransformer.
    This matches the Colab setup for consistent retrieval.
    """
    print(f"\n🗄️  Building ChromaDB (BGE-M3 via SentenceTransformer) at {db_path}...")

    # Load text
    print(f"📖 Loading text from {text_file_path}...")
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    print(f"✅ Loaded {len(full_text)} characters")

    # Chunking for retrieval - SAME AS COLAB
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        separators=["\n\n\n", "\n\n", "\n", "۔", "."]
    )
    chunks = text_splitter.split_text(full_text)
    print(f"📦 Created {len(chunks)} chunks")

    # Init Chroma with BGE-M3 via SentenceTransformer (same as Colab)
    print("🔧 Loading BGE-M3 embeddings via SentenceTransformer...")
    client = chromadb.PersistentClient(path=db_path)
    
    bge_m3_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3",
        device="cpu"  # PyTorch not compiled with CUDA
    )

    # Delete old collection if exists
    try:
        client.delete_collection("tashreeh_chroma_bgem3")
        print("🗑️  Deleted old collection")
    except:
        pass

    # Create collection
    collection = client.create_collection(
        name="tashreeh_chroma_bgem3",
        embedding_function=bge_m3_embedder,
        metadata={"description": "Bang-e-Dara Tashreeh using BGE-M3 embeddings"}
    )
    print("✅ Collection created")

    # Add chunks to ChromaDB
    print("💾 Adding chunks to ChromaDB...")
    batch_size = 10  # Smaller batches to avoid timeout

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
        batch_metadatas = [{
            "source": "Tashreeh Bang-e-Dara",
            "chunk_id": j,
            "chunk_size": len(chunk)
        } for j, chunk in enumerate(batch_chunks, start=i)]

        # Remove empty chunks
        valid_indices = [idx for idx, chunk in enumerate(batch_chunks) if chunk.strip()]

        if valid_indices:
            collection.add(
                documents=[batch_chunks[idx] for idx in valid_indices],
                ids=[batch_ids[idx] for idx in valid_indices],
                metadatas=[batch_metadatas[idx] for idx in valid_indices]
            )
            print(f"   ✅ Added batch {i//batch_size + 1}: {len(valid_indices)} chunks")

    print(f"\n🎉 Done! Saved {len(chunks)} chunks to ChromaDB using BGE-M3 embeddings.")
    print(f"📊 Total documents in collection: {collection.count()}")
    return collection


if __name__ == "__main__":
    collection = build_chromadb()

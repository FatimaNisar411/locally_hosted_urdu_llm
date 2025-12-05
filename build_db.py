# """
# Build ChromaDB collection with BGE-M3 embeddings for Bang-e-Dara Tashreeh
# """
# import chromadb
# from chromadb.utils import embedding_functions
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def build_chromadb(text_file_path="tashreeh_sample.txt", db_path="db"):
#     """
#     Save extracted poem + explanation text to ChromaDB using BGE-M3 embeddings via SentenceTransformer.
#     This matches the Colab setup for consistent retrieval.
#     """
#     print(f"\n🗄️  Building ChromaDB (BGE-M3 via SentenceTransformer) at {db_path}...")

#     # Load text
#     print(f"📖 Loading text from {text_file_path}...")
#     with open(text_file_path, "r", encoding="utf-8") as f:
#         full_text = f.read()
#     print(f"✅ Loaded {len(full_text)} characters")

#     # Chunking for retrieval - SAME AS COLAB
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=5000,
#         chunk_overlap=1200,
#         separators=["\n\n\n", "\n\n", "\n", "۔", "."]
#     )
#     chunks = text_splitter.split_text(full_text)
#     print(f"📦 Created {len(chunks)} chunks")

#     # Init Chroma with BGE-M3 via SentenceTransformer (same as Colab)
#     print("🔧 Loading BGE-M3 embeddings via SentenceTransformer...")
#     client = chromadb.PersistentClient(path=db_path)
    
#     bge_m3_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="BAAI/bge-m3",
#         device="cpu"  # PyTorch not compiled with CUDA
#     )

#     # Delete old collection if exists
#     try:
#         client.delete_collection("tashreeh_chroma_bgem3")
#         print("🗑️  Deleted old collection")
#     except:
#         pass

#     # Create collection
#     collection = client.create_collection(
#         name="tashreeh_chroma_bgem3",
#         embedding_function=bge_m3_embedder,
#         metadata={"description": "Bang-e-Dara Tashreeh using BGE-M3 embeddings"}
#     )
#     print("✅ Collection created")

#     # Add chunks to ChromaDB
#     print("💾 Adding chunks to ChromaDB...")
#     batch_size = 10  # Smaller batches to avoid timeout

#     for i in range(0, len(chunks), batch_size):
#         batch_chunks = chunks[i:i+batch_size]
#         batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
#         batch_metadatas = [{
#             "source": "Tashreeh Bang-e-Dara",
#             "chunk_id": j,
#             "chunk_size": len(chunk)
#         } for j, chunk in enumerate(batch_chunks, start=i)]

#         # Remove empty chunks
#         valid_indices = [idx for idx, chunk in enumerate(batch_chunks) if chunk.strip()]

#         if valid_indices:
#             collection.add(
#                 documents=[batch_chunks[idx] for idx in valid_indices],
#                 ids=[batch_ids[idx] for idx in valid_indices],
#                 metadatas=[batch_metadatas[idx] for idx in valid_indices]
#             )
#             print(f"   ✅ Added batch {i//batch_size + 1}: {len(valid_indices)} chunks")

#     print(f"\n🎉 Done! Saved {len(chunks)} chunks to ChromaDB using BGE-M3 embeddings.")
#     print(f"📊 Total documents in collection: {collection.count()}")
#     return collection


# if __name__ == "__main__":
#     collection = build_chromadb()






# import os
# import re
# from chromadb import PersistentClient
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # --- CONFIG ---
# DATA_FILE = "tashreeh_sample.txt"
# DB_DIR = "db"
# COLLECTION_NAME = "bang_e_dara"
# EMBEDDING_MODEL = "BAAI/bge-m3"
# CHUNK_SIZE = 3500
# CHUNK_OVERLAP = 350

# # --- HARDCODED FIRST 10 POEM NAMES ---
# POEM_TITLES = POEM_TITLES = [
#     "ہالہ",
#     "گل رنگیں",
#     "عہد طفلی",
#     "مرزا غالب",
#     "ابر کوہسار",
#     "ایک مکڑا اور مکھی",
#     "ایک پہاڑاورگلہری",
#     "ایک گائے اور بکری",
#     "بچے کی دعا",
#     "ہمدردی"
# ]

# def find_poem_ranges(text, poem_titles):
#     # Find the first occurrence of each poem title
#     starts = []
#     for title in poem_titles:
#         match = re.search(re.escape(title), text)
#         if match:
#             starts.append((match.start(), title))
#     starts.sort()
#     # Build (start, end, title) ranges
#     ranges = []
#     for i, (start_idx, title) in enumerate(starts):
#         end_idx = starts[i+1][0] if i+1 < len(starts) else len(text)
#         ranges.append((start_idx, end_idx, title))
#     return ranges

# def assign_poem_to_chunk(chunk_start, poem_ranges):
#     for start, end, title in poem_ranges:
#         if start <= chunk_start < end:
#             return title
#     return None

# def build_chromadb():
#     # Read the file
#     with open(DATA_FILE, encoding="utf-8") as f:
#         text = f.read()

#     # Debug: Print lines containing any poem title
#     print("\n--- Poem Title Matching Debug ---")
#     for title in POEM_TITLES:
#         matches = [line for line in text.splitlines() if title in line]
#         print(f"Title: {title} | Matches: {matches}")
#     print("--- End Debug ---\n")

#     # Find poem ranges in the main text
#     poem_ranges = find_poem_ranges(text, POEM_TITLES)
#     if not poem_ranges:
#         print("No poem ranges found! Check poem titles and text formatting.")
#         return
#     print(f"Found {len(poem_ranges)} poem ranges in text.")

#     # Chunk the text
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
#     )
#     chunks = []
#     metadatas = []
#     for chunk in splitter.split_text(text):
#         # Find the start index of this chunk in the text
#         chunk_start = text.find(chunk)
#         poem = assign_poem_to_chunk(chunk_start, poem_ranges)
#         if poem is None:
#             poem = "Unknown"
#         metadatas.append({
#             "poem": poem,
#             "start": chunk_start,
#         })
#         chunks.append(chunk)

#     # Embedding
#     model = SentenceTransformer(EMBEDDING_MODEL)
#     embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)

#     # ChromaDB
#     client = PersistentClient(path=DB_DIR, settings=Settings(allow_reset=True))
#     if COLLECTION_NAME in [c.name for c in client.list_collections()]:
#         client.delete_collection(COLLECTION_NAME)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     # Add to collection
#     ids = [f"chunk_{i}" for i in range(len(chunks))]
#     collection.add(
#         documents=chunks,
#         embeddings=embeddings.tolist(),
#         metadatas=metadatas,
#         ids=ids,
#     )
#     print(f"Added {len(chunks)} chunks to collection '{COLLECTION_NAME}'.")

# if __name__ == "__main__":
#     build_chromadb()
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG ---
DATA_FILE = "tashreeh_sample.txt"
DB_DIR = "db"
COLLECTION_NAME = "tashreeh_chroma_bgem3"  # ← SAME AS OLD CODE
EMBED_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 3500
CHUNK_OVERLAP = 350

POEM_TITLES = [
    "ہالہ", "گل رنگیں", "عہد طفلی", "مرزا غالب", "ابر کوہسار",
    "ایک مکڑا اور مکھی", "ایک پہاڑاورگلہری", "ایک گائے اور بکری",
    "بچے کی دعا", "ہمدردی"
]

def find_poem_ranges(text, poem_titles):
    """Find start and end indices of each poem."""
    starts = []
    for title in poem_titles:
        match = re.search(re.escape(title), text)
        if match:
            starts.append((match.start(), title))
    starts.sort()
    
    ranges = []
    for i, (start, title) in enumerate(starts):
        end = starts[i+1][0] if i+1 < len(starts) else len(text)
        ranges.append((start, end, title))
    return ranges

def build_chromadb():
    # Load full text
    with open(DATA_FILE, encoding="utf-8") as f:
        text = f.read()
    print(f"📖 Loaded text ({len(text)} chars)")

    # Find poem ranges
    poem_ranges = find_poem_ranges(text, POEM_TITLES)
    print(f"📌 Found {len(poem_ranges)} poem ranges")

    # Split per poem
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    metadatas = []

    for start, end, title in poem_ranges:
        poem_text = text[start:end]
        poem_chunks = splitter.split_text(poem_text)
        for chunk in poem_chunks:
            chunk_start = text.find(chunk)
            chunks.append(chunk)
            metadatas.append({
                "poem": title,
                "start": chunk_start,
                "chunk_size": len(chunk)
            })

    print(f"📦 Created {len(chunks)} chunks (poem-wise)")

    # -------------------------------------------
    # EXACT OLD LOGIC FOR COLLECTION
    # -------------------------------------------
    print("🔧 Loading BGE-M3 embedding function (same as old code)...")

    bge_m3_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete old collection exactly like before
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Deleted old collection")
    except:
        pass

    # Create collection same as before
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=bge_m3_embedder,
        metadata={"description": "Bang-e-Dara with poem metadata using BGE-M3"}
    )
    print("✅ Collection created")

    # Add in batches
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i + len(batch_chunks))]

        # remove empty chunks
        valid_indices = [idx for idx, c in enumerate(batch_chunks) if c.strip()]

        if valid_indices:
            collection.add(
                documents=[batch_chunks[idx] for idx in valid_indices],
                metadatas=[batch_metadatas[idx] for idx in valid_indices],
                ids=[batch_ids[idx] for idx in valid_indices],
            )

        print(f"   ➕ Batch {i//batch_size+1} added")

    print(f"\n🎉 DONE! Saved {len(chunks)} chunks.")
    print(f"📊 Total documents: {collection.count()}")

    return collection


if __name__ == "__main__":
    build_chromadb()

# import os
# import re
# import chromadb
# from chromadb.utils import embedding_functions
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # --- CONFIG ---
# DATA_FILE = "tashreeh_sample.txt"
# DB_DIR = "db"
# COLLECTION_NAME = "tashreeh_chroma_bgem3"  # ← SAME AS OLD CODE
# EMBED_MODEL = "BAAI/bge-m3"

# CHUNK_SIZE = 3500
# CHUNK_OVERLAP = 350

# POEM_TITLES = [
#     "ہالہ", "گل رنگیں", "عہد طفلی", "مرزا غالب", "ابر کوہسار",
#     "ایک مکڑا اور مکھی", "ایک پہاڑاورگلہری", "ایک گائے اور بکری",
#     "بچے کی دعا", "ہمدردی"
# ]

# def find_poem_ranges(text, poem_titles):
#     starts = []
#     for title in poem_titles:
#         match = re.search(re.escape(title), text)
#         if match:
#             starts.append((match.start(), title))

#     starts.sort()
#     ranges = []
#     for i, (start, title) in enumerate(starts):
#         end = starts[i+1][0] if i+1 < len(starts) else len(text)
#         ranges.append((start, end, title))

#     return ranges


# def assign_poem_to_chunk(chunk_start, poem_ranges):
#     for start, end, title in poem_ranges:
#         if start <= chunk_start < end:
#             return title
#     return "Unknown"


# def build_chromadb():

#     # Load text
#     with open(DATA_FILE, encoding="utf-8") as f:
#         text = f.read()
#     print(f"📖 Loaded text ({len(text)} chars)")

#     # Find poem sections
#     poem_ranges = find_poem_ranges(text, POEM_TITLES)
#     print(f"📌 Found {len(poem_ranges)} poem ranges")

#     # Chunk text
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP
#     )

#     raw_chunks = splitter.split_text(text)

#     chunks = []
#     metadatas = []

#     for chunk in raw_chunks:
#         start_index = text.find(chunk)
#         poem = assign_poem_to_chunk(start_index, poem_ranges)

#         chunks.append(chunk)
#         metadatas.append({
#             "poem": poem,
#             "start": start_index,
#             "chunk_size": len(chunk),
#         })

#     print(f"📦 Created {len(chunks)} chunks")

#     # -------------------------------------------
#     # EXACT OLD LOGIC STARTS HERE
#     # -------------------------------------------

#     print("🔧 Loading BGE-M3 embedding function (same as old code)...")

#     bge_m3_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=EMBED_MODEL,
#         device="cpu"
#     )

#     client = chromadb.PersistentClient(path=DB_DIR)

#     # Delete old collection exactly like before
#     try:
#         client.delete_collection(COLLECTION_NAME)
#         print("🗑️ Deleted old collection")
#     except:
#         pass

#     # Create collection same as before
#     collection = client.create_collection(
#         name=COLLECTION_NAME,
#         embedding_function=bge_m3_embedder,
#         metadata={"description": "Bang-e-Dara with poem metadata using BGE-M3"}
#     )

#     print("✅ Collection created")

#     # OLD STYLE: batched uploading
#     batch_size = 10

#     for i in range(0, len(chunks), batch_size):
#         batch_chunks = chunks[i:i+batch_size]
#         batch_metadatas = metadatas[i:i+batch_size]
#         batch_ids = [f"chunk_{j}" for j in range(i, i + len(batch_chunks))]

#         # remove empty chunks
#         valid_indices = [idx for idx, c in enumerate(batch_chunks) if c.strip()]

#         if valid_indices:
#             collection.add(
#                 documents=[batch_chunks[idx] for idx in valid_indices],
#                 metadatas=[batch_metadatas[idx] for idx in valid_indices],
#                 ids=[batch_ids[idx] for idx in valid_indices],
#             )

#         print(f"   ➕ Batch {i//batch_size+1} added")

#     print(f"\n🎉 DONE! Saved {len(chunks)} chunks.")
#     print(f"📊 Total documents: {collection.count()}")

#     return collection


# if __name__ == "__main__":
#     build_chromadb()

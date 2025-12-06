from collections import Counter

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


# ------------------------------------------------
# 4. Example usage (uncomment)
# ------------------------------------------------

from chromadb import PersistentClient

client = PersistentClient(path="db")
collection = client.get_collection("tashreeh_chroma_bgem3")

# test_couplet = "چومتا ہے تیری پیشانی کو جھک کر اسماں، اے ممالہ! اے فصیلِ کشورِ ہندوستاں"
test_couplet = "اے گل رنگیں ترے پہلو میں شاید دل نہیں"

result = retrieve_poem(test_couplet, collection)

print("Title:", result["title"])
print("----")
for c in result["chunks"]:
    print(c[:200])  # show first 200 chars
    print("----")
results = collection.get(include=["metadatas"])
print(results["metadatas"])

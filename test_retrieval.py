"""
Test retrieval for a specific verse from Bang-e-Dara
"""
import chromadb

# Query text
query = """چومتا ہے تیری پیشانی کو جھک کر اسماں
اے ممالہ! اے فصیل کشور ہندوستاں"""

print(f"🔍 Testing retrieval for:")
print(query)
print("\n" + "="*80 + "\n")

# Connect to database
client = chromadb.PersistentClient(path="db")
collection = client.get_collection(name="tashreeh_chroma_bgem3")

print(f"📊 Collection info:")
print(f"   - Name: {collection.name}")
print(f"   - Total documents: {collection.count()}")
print("\n" + "="*80 + "\n")

# Query the collection
results = collection.query(
    query_texts=[query],
    n_results=5
)

# Display results
print(f"📦 Retrieved {len(results['documents'][0])} chunks:\n")

for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
    print(f"Result {i}:")
    print(f"Distance: {distance:.4f}")
    print(f"Content preview (first 500 chars):")
    print(doc[:500])
    print("\n" + "-"*80 + "\n")

# Show which is the best match
best_idx = results['distances'][0].index(min(results['distances'][0]))
print(f"✅ Best match is Result {best_idx + 1} with distance {results['distances'][0][best_idx]:.4f}")
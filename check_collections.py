import chromadb

# Check all database directories
for db_path in ["db", "dba", "dbc"]:
    try:
        print(f"\n=== Checking {db_path} ===")
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        if collections:
            for col in collections:
                print(f"  - {col.name}")
        else:
            print("  (no collections)")
    except Exception as e:
        print(f"  Error: {e}")

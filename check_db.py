import sqlite3

conn = sqlite3.connect('db/chroma.sqlite3')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", tables)

# Check collections table
try:
    cursor.execute("SELECT * FROM collections")
    collections = cursor.fetchall()
    print("\nCollections:")
    for col in collections:
        print(col)
except Exception as e:
    print(f"Error reading collections: {e}")

conn.close()

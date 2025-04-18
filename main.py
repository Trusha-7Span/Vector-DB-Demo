import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up Chroma with new style client (no deprecated config)
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection(name="my_documents")

# Load documents from text file
data_path = "documents/knowledge.txt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing file at: {data_path}")

with open(data_path, "r", encoding="utf-8") as file:
    content = file.read()

# Split into chunks (optional: improve by adding better chunking logic later)
chunks = [content[i:i+500] for i in range(0, len(content), 500)]

# Convert to embeddings and add to Chroma
print("🔄 Adding data to vector database...")
for chunk in chunks:
    embedding = model.encode([chunk])[0]
    doc_id = str(uuid.uuid4())
    collection.add(
        documents=[chunk],
        embeddings=[embedding.tolist()],
        ids=[doc_id]
    )
print("✅ Data added to Chroma vector store.")

# Start the query loop
while True:
    query = input("\n🔍 Ask a question (or type 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break

    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["documents", "distances"]
    )

    print("\n📚 Top Matches:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        confidence = (1 - distance) * 100  # cosine distance to confidence
        print(f"\n🔹 Match {i+1} ({confidence:.2f}% confidence):\n{doc.strip()[:300]}...")

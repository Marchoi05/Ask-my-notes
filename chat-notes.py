import ollama
from sentence_transformers import SentenceTransformer, util

# --- Load the search model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Read your notes from the file ---
with open("notes.txt", encoding="utf-8") as f:
    text = f.read()

# Chop by empty lines (paragraphs)
chunks = [c.strip() for c in text.split("\n\n") if c.strip() != ""]
print(f"Loaded {len(chunks)} chunks from your notes.")

# Turn all chunks into "meaning numbers" ONCE, at the start
chunk_embeddings = model.encode(chunks)

# --- Keep asking questions until you type 'quit' ---
while True:
    question = input("\nAsk a question (or type 'quit'): ")
    if question.lower() == "quit":
        break

    # Find the best 3 chunks (or fewer if you have fewer chunks)
    question_embedding = model.encode(question)
    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top = similarities.topk(k=min(3, len(chunks)))
    context = "\n\n".join(chunks[i] for i in top.indices)

    prompt = f"""Answer the question using ONLY the notes below.
If the answer is not in the notes, say "I can't find that in the notes."

NOTES:
{context}

QUESTION: {question}"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )
    print("\nANSWER:", response["message"]["content"])
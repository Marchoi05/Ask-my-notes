import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("notes.txt", encoding="utf-8") as f:
    text = f.read()
chunks = [c.strip() for c in text.split("\n\n") if c.strip() != ""]
chunk_embeddings = model.encode(chunks)

with open("questions.json", encoding="utf-8") as f:
    tests = json.load(f)

K = 1  # how many paragraphs the app looks at
points = 0

for t in tests:
    q_emb = model.encode(t["question"])
    sims = util.cos_sim(q_emb, chunk_embeddings)[0]
    top = sims.topk(k=min(K, len(chunks)))
    found = " ".join(chunks[i] for i in top.indices).lower()

    if t["keyword"].lower() in found:
        points += 1
        print(f"PASS: {t['question']}")
    else:
        print(f"FAIL: {t['question']}")

print(f"\nScore: {points} / {len(tests)}")
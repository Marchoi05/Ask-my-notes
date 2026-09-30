import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("Ask My Notes")

# Load the model and notes only ONCE (otherwise the page reloads them every click)
@st.cache_resource
def load_everything():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open("notes.txt", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip() != ""]
    embeddings = model.encode(chunks)
    return model, chunks, embeddings

model, chunks, chunk_embeddings = load_everything()
st.write(f"Loaded {len(chunks)} chunks from your notes.")

# The text box where you type
question = st.text_input("Ask a question about your notes:")

if question:
    question_embedding = model.encode(question)
    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top = similarities.topk(k=min(3, len(chunks)))
    best_chunks = [chunks[i] for i in top.indices]
    context = "\n\n".join(best_chunks)

    prompt = f"""Answer the question using ONLY the notes below.
If the answer is not in the notes, say "I can't find that in the notes."

NOTES:
{context}

QUESTION: {question}"""

    with st.spinner("Thinking..."):
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
        )

    st.subheader("Answer")
    st.write(response["message"]["content"])

    # Show WHERE the answer came from (this makes it trustworthy)
    with st.expander("Show the notes I used"):
        for c in best_chunks:
            st.write(c)
            st.divider()
import streamlit as st
import json
import numpy as np
import faiss
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load Q&A data
qa_data = []
with open("vehicle-bot/vehicles_qa_text.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        qa_data.append(json.loads(line.strip()))

texts = [
    f"Q: {item['text'][0]['content']}\nA: {item['text'][1]['content']}"
    for item in qa_data if item.get("text") and len(item["text"]) == 2
]

# Embedding and indexing
embedder = SentenceTransformer("intfloat/e5-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = embedder.encode(texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve_similar(query, top_k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    candidate_texts = [texts[i] for i in I[0]]
    scores = reranker.predict([(query, text) for text in candidate_texts])
    reranked = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in reranked[:3]]

# Load LLM
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

def is_safe_input(text):
    banned_keywords = [
        "rape", "kill", "hate", "nazi", "terrorist", "shoot", "suicide",
        "gay", "sex", "naked", "murder", "pedophile", "assault", "abuse"
    ]
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in banned_keywords) + r')\b', re.IGNORECASE)
    return not pattern.search(text)

def safe_generate_response(query):
    if not is_safe_input(query):
        return "❌ Please keep your question respectful and appropriate."

    context = "\n\n".join(retrieve_similar(query))

    prompt = f"""
You are a highly accurate vehicle expert assistant. Respond ONLY using the information in the context below.

Do not guess, assume, or add anything extra. Stick to facts. Return clear and concise answers only.

---
{context}
---

Question: {query}
Answer:"""

    result = llm(prompt, max_new_tokens=150, do_sample=False, temperature=0.4)
    output = result[0]["generated_text"]
    answer = output.split("Answer:")[-1].strip()
    return answer, context

# Streamlit UI
st.set_page_config(page_title="BikePedia Q&A Bot", page_icon="🏍️")
st.title("🏍️ BikePedia Chatbot")
st.write("Ask me about specs, mileage, price, etc.")

query = st.text_input("Type your question here:")

if query:
    with st.spinner("Generating answer..."):
        answer, context = safe_generate_response(query)
        st.markdown("### 🤖 Answer")
        st.success(answer)
        with st.expander("📚 Retrieved Context"):
            st.code(context)

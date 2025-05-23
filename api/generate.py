# api/generate.py
from fastapi import FastAPI
from pydantic import BaseModel
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Load and index your data on cold start ---
with open("vehicles_qa_text.jsonl", "r", encoding="utf-8") as f:
    qa_data = [json.loads(line) for line in f]

texts = [
    f"Q: {item['text'][0]['content']}\nA: {item['text'][1]['content']}"
    for item in qa_data if item.get("text") and len(item["text"]) == 2
]

embedder = SentenceTransformer("intfloat/e5-base-v2")
embeddings = embedder.encode(texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

def retrieve_similar(query, k=3):
    vec = embedder.encode([query])
    D,I = index.search(np.array(vec), k)
    return [texts[i] for i in I[0]]

def generate_answer(query: str) -> str:
    ctx = "\n\n".join(retrieve_similar(query))
    prompt = f"You are a vehicle expert. Answer only using the context below.\n\n{ctx}\n\nQuestion: {query}\nAnswer:"
    out = llm(prompt, max_new_tokens=150, do_sample=False, temperature=0.4)[0]["generated_text"]
    return out.split("Answer:")[-1].strip()

# --- FastAPI setup ---
app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/generate")
def generate(q: Query):
    return {"answer": generate_answer(q.query)}

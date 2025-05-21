# bikepedia-rag-chatbot
🚗 A Streamlit-powered Q&amp;A chatbot for vehicle specifications, prices, and features using Retrieval-Augmented Generation (RAG) and Microsoft’s Phi-2 LLM. Powered by semantic search (FAISS + sentence-transformers) and hosted for free via Streamlit Cloud.

bikepedia-rag-chatbot/
├── app.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
└── vehicles_qa_text.jsonl   # Your knowledge base

🌐 What This Repo Does
Embeds your custom Q&A dataset (vehicles_qa_text.jsonl)

Indexes with FAISS for fast semantic search

Uses a reranker for accuracy (cross-encoder)

Passes top results to phi-2 LLM for clean, contextual answers

Displays a Streamlit chatbot with:

🚀 Instant Q&A

📚 Expandable context

✅ Respect filter for safety


#streamlit #rag #chatbot #llm #vehicle #qa #gradio #transformers #phi2 #semanticsearch #open-source

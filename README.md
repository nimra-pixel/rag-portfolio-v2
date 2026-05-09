# 🧠 AI/ML Knowledge Base — RAG App v2 (Zero Cost)

> **Why v2?** My [first RAG app](https://github.com/nimra-pixel/rag-portfolio) used the OpenAI API. After deploying it publicly I realised anyone visiting the live demo would hit a rate-limit error without API credits — not great for a portfolio. So I rebuilt it using the **HuggingFace free inference API**: same hybrid retrieval architecture, same citation enforcement, zero cost to run for any visitor.

🚀 **[Live Demo →](https://your-v2-app.streamlit.app)**  
🔗 **[v1 (OpenAI version) →](https://github.com/nimra-pixel/rag-portfolio)**

---

## What changed from v1

| | v1 (OpenAI) | v2 (HuggingFace) |
|---|---|---|
| Embeddings | text-embedding-3-small | all-MiniLM-L6-v2 ✅ free |
| Generation | GPT-4o-mini | Zephyr-7B ✅ free |
| API cost | ~$0.002/query | $0.00 |
| Works publicly | ❌ fails without credits | ✅ always works |
| Local fallback | ❌ | ✅ retrieval-only mode |

## What stayed the same

- Hybrid BM25 + vector retrieval
- Reciprocal Rank Fusion (RRF)
- Citation-enforced answers
- Live pipeline trace
- 10 AI/ML topics indexed

## Architecture

```
Query → BM25 sparse search ┐
                            ├→ RRF fusion → Top-k chunks → Zephyr-7B → Answer + citations
Query → MiniLM vectors     ┘
```

## Run locally

```bash
git clone https://github.com/nimra-pixel/rag-portfolio-v2
cd rag-portfolio-v2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# No API key needed — works out of the box
```

## Stack

`Python` `Streamlit` `HuggingFace Inference API` `all-MiniLM-L6-v2` `Zephyr-7B` `NumPy` `BM25`

---

Built by [Nimra](https://linkedin.com/in/yourprofile)  

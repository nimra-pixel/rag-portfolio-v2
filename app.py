import streamlit as st
import numpy as np
import re
import requests
from typing import List, Dict

st.set_page_config(
    page_title="AI/ML Knowledge Base v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

KNOWLEDGE_BASE = [
    {"id": 0, "topic": "Transformer Architecture", "content": "The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), revolutionised NLP by replacing recurrence with self-attention mechanisms. The core idea is that every token in a sequence can attend to every other token in parallel, enabling the model to capture long-range dependencies efficiently. The architecture consists of an encoder and decoder, each made of stacked layers containing multi-head self-attention and feed-forward networks. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Positional encodings are added to token embeddings to inject sequence order information. Layer normalisation and residual connections stabilise training. The Transformer became the foundation for BERT, GPT, T5, and virtually every modern large language model."},
    {"id": 1, "topic": "Retrieval-Augmented Generation (RAG)", "content": "Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), combines parametric knowledge stored in model weights with non-parametric knowledge retrieved from an external corpus at inference time. The pipeline has two phases: retrieval and generation. During retrieval, a dense vector search finds the top-k most relevant documents for a query using embedding similarity. During generation, those documents are prepended to the prompt as context, grounding the model response. RAG solves two key problems: knowledge cutoffs and hallucination. Hybrid RAG combines dense retrieval with sparse retrieval (BM25) and uses Reciprocal Rank Fusion to merge results. Production RAG systems measure quality with RAGAS metrics: faithfulness, answer relevancy, and context recall."},
    {"id": 2, "topic": "Large Language Models (LLMs)", "content": "Large Language Models are neural networks trained on massive text corpora using the self-supervised objective of next-token prediction. GPT-3 (175B parameters) demonstrated that scale alone could produce emergent capabilities including few-shot learning. Modern LLMs like GPT-4, Claude, Gemini, and Llama are trained in stages: pre-training on internet-scale text, supervised fine-tuning (SFT) on instruction-following examples, and RLHF to align outputs with human preferences. Key concepts include context window, temperature (controls output randomness), and tokenisation. Inference efficiency techniques include KV-caching, flash attention, quantisation (INT8/INT4), and speculative decoding."},
    {"id": 3, "topic": "Fine-Tuning & LoRA", "content": "Fine-tuning adapts a pre-trained model to a specific task by continuing training on a smaller task-specific dataset. Full fine-tuning updates all model weights and is computationally expensive. LoRA (Low-Rank Adaptation) freezes pre-trained weights and injects trainable low-rank decomposition matrices into each transformer layer, reducing trainable parameters by 10000x while matching full fine-tuning performance. QLoRA extends LoRA with 4-bit quantisation, enabling fine-tuning of 70B models on a single GPU. DPO (Direct Preference Optimisation) fine-tunes models on preference pairs without a separate reward model."},
    {"id": 4, "topic": "Embeddings & Vector Search", "content": "Embeddings are dense vector representations of text that capture semantic meaning. Similar meanings cluster together in vector space. Models like Sentence-BERT and E5 encode sentences into fixed-size vectors (384-1536 dimensions). Cosine similarity measures angle between vectors, returning 1 for identical meaning and 0 for orthogonal. Vector databases (FAISS, Pinecone, Weaviate, Chroma) enable approximate nearest-neighbour search using HNSW and IVF algorithms. FAISS is an open-source library optimised for billion-scale similarity search. The key tradeoff is recall vs latency: HNSW is fast but approximate; flat search is exact but slow."},
    {"id": 5, "topic": "Attention Mechanism", "content": "Attention allows a model to focus on relevant parts of the input when producing each output token. Scaled dot-product attention computes: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Multi-head attention runs h parallel attention heads each learning different relational patterns. Self-attention means Q, K, V all come from the same sequence. Causal masked self-attention prevents tokens from attending to future positions, essential for autoregressive generation. Flash Attention (Dao et al., 2022) reduces memory from O(N squared) to O(N) by tiling computations, enabling training on much longer sequences."},
    {"id": 6, "topic": "Evaluation Metrics for LLMs", "content": "Evaluating LLMs is hard because natural language has no single correct answer. Automatic metrics include BLEU (n-gram overlap), ROUGE (for summarisation), BERTScore (semantic similarity), and perplexity. RAGAS is a RAG-specific evaluation framework measuring Faithfulness (does the answer use only retrieved context?), Answer Relevancy (does it address the question?), and Context Recall (did retrieval find the needed chunks?). LLM-as-judge uses a strong model to score outputs on rubrics. Human evaluation remains the gold standard but is expensive. MT-Bench and MMLU are popular general capability benchmarks."},
    {"id": 7, "topic": "Prompt Engineering", "content": "Prompt engineering is the practice of designing inputs to LLMs to elicit desired outputs. Zero-shot prompting gives the model a task with no examples. Few-shot prompting includes 2-5 examples in the prompt. Chain-of-thought (CoT) prompting instructs the model to reason step by step, improving performance on arithmetic and logical tasks. Structured output prompting uses JSON schemas or XML tags to enforce output format. System prompts set model behaviour and persona. Temperature controls randomness: 0 for deterministic tasks, 0.7-1.0 for creative tasks. Top-p nucleus sampling samples from tokens whose cumulative probability exceeds p."},
    {"id": 8, "topic": "RLHF & Alignment", "content": "RLHF aligns language models with human preferences through three stages: supervised fine-tuning (SFT) on demonstrations, reward model training on human preference rankings, and RL optimisation using PPO with a KL-divergence penalty. InstructGPT showed that a 1.3B RLHF model was preferred over a 175B base model. Constitutional AI extends RLHF with an AI-generated critique-revision loop guided by a set of principles. DPO simplifies RLHF by directly optimising on chosen vs rejected preference pairs without a separate reward model, using a closed-form solution derived from the Bradley-Terry preference model."},
    {"id": 9, "topic": "Quantisation & Inference Optimisation", "content": "Quantisation reduces model size and speeds up inference by representing weights in lower precision. FP32 to FP16/BF16 halves memory with minimal quality loss. INT8 further halves memory. INT4/NF4 used in QLoRA compresses a 7B model to about 3.5GB. Post-training quantisation (PTQ) quantises a trained model without retraining. KV-cache stores previously computed key-value pairs to avoid recomputation during autoregressive decoding. Speculative decoding uses a small draft model to propose tokens a larger model verifies in parallel, achieving 2-3x speedup. vLLM uses PagedAttention for efficient KV-cache management enabling high-throughput serving."},
]

st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1a2332 100%);border:1px solid #30363d;border-radius:16px;padding:2.5rem 2rem 2rem;margin-bottom:1rem;text-align:center}
.hero h1{font-size:2rem;font-weight:700;color:#f0f6fc;margin:0}
.hero p{color:#8b949e;margin:.5rem 0 0;font-size:.95rem}
.hero .badge{display:inline-block;background:#1f4e2e;color:#3fb950;font-size:.75rem;font-weight:600;padding:4px 12px;border-radius:20px;border:1px solid #3fb950;margin-top:.8rem}
.story-box{background:#161b22;border:1px solid #30363d;border-left:3px solid #f0883e;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1.2rem;font-size:.88rem;color:#8b949e;line-height:1.7}
.story-box strong{color:#f0f6fc}
.chunk-card{background:#161b22;border:1px solid #30363d;border-left:3px solid #58a6ff;border-radius:8px;padding:.9rem 1rem;margin-bottom:.6rem;font-size:.85rem;color:#c9d1d9;line-height:1.6}
.chunk-title{font-weight:600;color:#58a6ff;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem}
.score-pill{display:inline-block;background:#1c2d3f;color:#79c0ff;font-size:.72rem;padding:2px 8px;border-radius:20px;margin-bottom:.5rem}
.answer-box{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.2rem 1.4rem;color:#f0f6fc;font-size:.95rem;line-height:1.8;margin-bottom:1rem}
.tag{display:inline-block;background:#1c2d3f;color:#79c0ff;font-size:.72rem;padding:3px 10px;border-radius:20px;margin:3px}
.ps{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:6px;padding:4px 10px;font-size:.78rem;color:#8b949e;margin:2px}
.diff-row{display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap}
.diff-card{flex:1;min-width:140px;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:.9rem}
.diff-card h4{font-size:.78rem;color:#8b949e;margin:0 0 6px;text-transform:uppercase;letter-spacing:.05em}
.diff-card p{font-size:.85rem;color:#f0f6fc;margin:0}
.v1{border-left:3px solid #f85149}
.v2{border-left:3px solid #3fb950}
</style>
""", unsafe_allow_html=True)

# ── Local embeddings — no API needed ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed(texts: tuple) -> np.ndarray:
    model = load_model()
    return model.encode(list(texts), normalize_embeddings=True)

# ── LLM via HuggingFace (with local fallback) ─────────────────────────────────
HF_LLM = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

def hf_headers():
    t = st.secrets.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {t}"} if t else {}

def bm25(query, docs, k1=1.5, b=0.75):
    qt = query.lower().split()
    tok = [d.lower().split() for d in docs]
    avg = np.mean([len(d) for d in tok])
    s = np.zeros(len(docs))
    for term in qt:
        df = sum(1 for d in tok if term in d)
        if not df: continue
        idf = np.log((len(docs)-df+0.5)/(df+0.5)+1)
        for i, d in enumerate(tok):
            tf = d.count(term); dl = len(d)
            s[i] += idf*(tf*(k1+1))/(tf+k1*(1-b+b*dl/avg))
    return s

def retrieve(query, top_k=3):
    texts = tuple(c["content"] for c in KNOWLEDGE_BASE)
    all_e = embed(texts + (query,))
    de, qe = all_e[:-1], all_e[-1]
    vec = de @ qe          # already normalised → cosine similarity
    kw  = bm25(query, list(texts))
    rrf = np.zeros(len(texts))
    for r, i in enumerate(np.argsort(-vec)): rrf[i] += 1/(61+r)
    for r, i in enumerate(np.argsort(-kw)):  rrf[i] += 1/(61+r)
    out = []
    for i in np.argsort(-rrf)[:top_k]:
        c = KNOWLEDGE_BASE[i].copy()
        c["rrf"] = round(float(rrf[i]),4)
        c["cos"] = round(float(vec[i]),3)
        out.append(c)
    return out

def generate(query, chunks):
    ctx = "\n\n---\n\n".join(f"[SOURCE:{c['id']} | {c['topic']}]\n{c['content'][:400]}" for c in chunks)
    prompt = (f"<|system|>\nYou are an expert AI/ML educator. Answer using ONLY the provided context. "
              f"Cite every fact as [SOURCE:id]. Be concise and clear.\n</s>\n"
              f"<|user|>\nContext:\n{ctx}\n\nQuestion: {query}\n</s>\n<|assistant|>")
    try:
        r = requests.post(HF_LLM, headers=hf_headers(),
            json={"inputs": prompt,
                  "parameters": {"max_new_tokens": 300, "temperature": 0.3,
                                 "return_full_text": False, "do_sample": True},
                  "options": {"wait_for_model": True, "use_cache": False}},
            timeout=60)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                t = data[0].get("generated_text","").strip()
                if t: return t, "🤖 Zephyr-7B (HuggingFace)"
    except Exception:
        pass
    # Local fallback
    lines = ["**Answer built directly from retrieved chunks:**\n"]
    for c in chunks:
        sents = [s.strip() for s in c["content"].replace("\n"," ").split(".") if len(s.strip())>30]
        lines.append(f"**{c['topic']}** [SOURCE:{c['id']}]: {'. '.join(sents[:2])}.")
    return "\n\n".join(lines), "📚 Local fallback (retrieval-only mode)"

def cite(text, chunks):
    m = {str(c["id"]): c["topic"] for c in chunks}
    return re.sub(r'\[SOURCE:(\d+)\]', lambda x: f"**[{m.get(x.group(1),'src')}]**", text)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k         = st.slider("Chunks to retrieve", 1, 5, 3)
    show_chunks   = st.toggle("Show retrieved chunks", True)
    show_pipeline = st.toggle("Show pipeline trace", True)
    show_diff     = st.toggle("Show v1 vs v2 comparison", True)
    st.markdown("---")
    st.markdown("### 📚 Topics Indexed")
    for c in KNOWLEDGE_BASE:
        st.markdown(f"<span class='tag'>{c['topic']}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""**Stack**  
    Streamlit · all-MiniLM-L6-v2 (local)  
    Zephyr-7B (HuggingFace) · NumPy  
    BM25 + Vector · RRF fusion
    """)
    st.caption("Built by Nimra · [v1](https://github.com/nimra-pixel/rag-portfolio) · [v2](https://github.com/nimra-pixel/rag-portfolio-v2)")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 AI/ML Knowledge Base <span style="color:#58a6ff">v2</span></h1>
  <p>Ask anything about ML, LLMs, RAG, Fine-tuning, Transformers & more</p>
  <span class="badge">✅ 100% Free — No API Key Required</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="story-box">
  <strong>Why v2?</strong> — My first RAG app (v1) used the OpenAI API.
  After shipping it I realised <strong>not everyone has API credits</strong>, and a public portfolio demo
  that throws a rate-limit error the moment someone tries it defeats the purpose.
  So I rebuilt it: embeddings now run <strong>locally inside the app</strong> using all-MiniLM-L6-v2 —
  no external API calls, no tokens, no rate limits. Same hybrid retrieval architecture, zero cost.
  This is the engineering tradeoff: <em>accessibility vs convenience</em>.
</div>
""", unsafe_allow_html=True)

if show_diff:
    st.markdown("#### 🔄 v1 → v2: What changed and why")
    st.markdown("""
    <div class="diff-row">
      <div class="diff-card v1"><h4>v1 — OpenAI</h4><p>text-embedding-3-small + GPT-4o-mini<br><br>❌ Requires paid API key<br>❌ Fails publicly without credits<br>❌ Not accessible to everyone</p></div>
      <div class="diff-card v2"><h4>v2 — Local ✅</h4><p>all-MiniLM-L6-v2 runs inside app<br><br>✅ No API key ever needed<br>✅ Works for every visitor<br>✅ Graceful LLM fallback</p></div>
      <div class="diff-card v2"><h4>Same RAG core ✅</h4><p>Hybrid BM25 + Vector<br>Reciprocal Rank Fusion<br>Citation enforcement<br>Pipeline tracing</p></div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Topics", "10")
c2.metric("Embeddings", "Local MiniLM ✅")
c3.metric("Generation", "Zephyr-7B")
c4.metric("API cost", "$0.00")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**💡 Try asking:**")
suggestions = [
    "How does the Transformer architecture work?",
    "What is RAG and how does hybrid retrieval improve it?",
    "Explain LoRA and QLoRA",
    "What is RLHF and how does DPO differ?",
    "How does quantisation speed up LLM inference?",
]
cols = st.columns(len(suggestions))
for i, (col, q) in enumerate(zip(cols, suggestions)):
    with col:
        if st.button(q, key=f"s{i}", use_container_width=True):
            st.session_state["query"] = q

query = st.text_input("Ask a question",
    value=st.session_state.get("query",""),
    placeholder="e.g. What is the difference between LoRA and full fine-tuning?",
    label_visibility="collapsed")

if query:
    with st.spinner("Retrieving and generating..."):
        chunks = retrieve(query, top_k)
        answer, src_label = generate(query, chunks)

    if show_pipeline:
        st.markdown("**🔍 Pipeline trace**")
        st.markdown(
            f"<span class='ps'>Query</span> → "
            f"<span class='ps'>BM25 sparse</span> + "
            f"<span class='ps'>MiniLM local vectors</span> → "
            f"<span class='ps'>RRF fusion</span> → "
            f"<span class='ps'>Top {top_k} chunks</span> → "
            f"<span class='ps'>Zephyr-7B</span> → "
            f"<span class='ps'>Answer + citations</span>",
            unsafe_allow_html=True)
        st.caption(f"Generation: {src_label}")
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**🤖 Answer**")
    st.markdown(f'<div class="answer-box">{cite(answer, chunks)}</div>', unsafe_allow_html=True)

    if show_chunks:
        st.markdown(f"**📄 Retrieved chunks** (top {top_k} by RRF score)")
        for c in chunks:
            st.markdown(
                f'<div class="chunk-card">'
                f'<div class="chunk-title">{c["topic"]}</div>'
                f'<span class="score-pill">RRF: {c["rrf"]} · Cosine: {c["cos"]}</span>'
                f'<div>{c["content"][:300]}...</div>'
                f'</div>', unsafe_allow_html=True)

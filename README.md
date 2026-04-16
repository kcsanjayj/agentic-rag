# Agentic-RAG: Self-Correcting RAG System

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Live Demo](https://img.shields.io/badge/demo-live-green)
![License](https://img.shields.io/badge/license-MIT-blue)

🚀 Multi-agent RAG with evaluation gating  
📉 25% reduction in hallucinations (vs baseline RAG)  
📊 +21% relevance improvement  
⚡ 7 Cloud LLM providers (API-based, online only)  

💡 From basic RAG → self-correcting AI system

---

## Results

| Metric | Baseline | Agentic-RAG |
|--------|----------|-------------|
| Relevance | 7.1 | 8.6 |
| Hallucination | High | -25% |
| Iterations | 1 | Up to 3 |

Dataset: 50 QA (arXiv) | Evaluated using GPT-4 rubric (relevance, grounding, completeness)

---

## Demo

**Live Demo:** https://agentic-rag-gamma.vercel.app  
**Backend API:** https://agentic-rag-production.up.railway.app

> Upload document  
> Ask question  
> See evaluation score (e.g., 8.7/10) + execution trace

---

## Example Output

**Query:** What is attention mechanism?

**Answer (Agentic-RAG):**  
Attention allows models to dynamically weight token importance, enabling parallel processing of sequences (Vaswani et al., 2017).

**Score:** 8.7 / 10  
**Iterations:** 2  
**Sources:** `attention-is-all-you-need.pdf`

---

## Why This Matters

Most RAG systems:
- ❌ No validation
- ❌ Hallucinate

This system:
- ✅ Evaluates responses
- ✅ Self-corrects
- ✅ Blocks low-quality answers

---

## How It Works

Query → Retrieval → Generation → Critic → Evaluation → Final Answer

---

## Architecture

```
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌──────────┐   ┌─────────────┐
│  User   │ → │ Retrieval│ → │   LLM    │ → │ Critic │ → │ Evaluation│ → │ Final Answer│
└─────────┘   └──────────┘   └──────────┘   └────────┘   └──────────┘   └─────────────┘
```

---

## Key Features

- Multi-agent self-correction loop
- Evaluation gating
- Hybrid retrieval (dense + sparse)
- Execution trace
- UI-based API key configuration
- Runtime configuration (no .env file needed)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| Vector DB | ChromaDB |
| LLMs | OpenAI, Gemini, Claude, NVIDIA, Groq, HuggingFace (Cloud APIs - keys required) |
| Embeddings | sentence-transformers |
| Frontend | Vanilla JS |

---

## Quick Start

```bash
git clone https://github.com/kcsanjayj/agentic-rag.git
cd agentic-rag
pip install -r requirements.txt
python start.py
```

Open http://localhost:8000 → Click **AI Config** → Enter your API key → Select provider → Start chatting.

> ⚠️ **Online Only**: This system requires API keys for cloud LLM providers. No local model support.

---

## Documentation

- Architecture → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)  
- API Reference → [docs/API.md](docs/API.md)  
- Evaluation → [docs/EVALUATION.md](docs/EVALUATION.md)  
- Deployment → [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)  
- Edge Cases → [docs/FAILURES.md](docs/FAILURES.md)

---

## ⭐ Why This Project Stands Out

- Demonstrates **production-grade RAG design**
- Implements **evaluation + self-correction** (rare in projects)
- Focuses on **reliability, not just generation**

👉 Built to solve real-world LLM problems, not just demos

---

## License

MIT – see [LICENSE](LICENSE)

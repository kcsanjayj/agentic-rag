# ⚡ Aetherion — Evaluation-Driven Agentic RAG

> **A self-evaluating RAG system that retrieves evidence, generates answers, critiques them, and automatically refines weak responses before finalization.**

---

## ⚡ 30-Second Overview

**Aetherion is not a single LLM call.**

It adds an evaluation and refinement loop on top of traditional RAG:

```text id="n1h3k8"
User Query
    │
    ▼
  Planner
    │
    ▼
 Retriever
    │
    ▼
   LLM
    │
    ▼
  Critic
   │  │
   │  └── Weak → Retry → Refine
   │                  │
   └──── Good ◄──────┘
          │
          ▼
     Final Answer
```

### 🧠 What I built

* 🧠 Query planning
* 📚 Retrieval-Augmented Generation
* 🤖 Multi-LLM orchestration
* 🔍 Critic-based evaluation
* 🔄 Bounded self-correction
* ♻️ Retry/refinement loop
* 🛡️ Provider fallback
* 📊 Execution tracing
* ⚡ Async FastAPI backend

---

# 🎥 Live Demo

**Try Aetherion:**
https://agentic-rag-gamma.vercel.app

---

# 🏗️ Architecture

```text id="3f9k1x"
                  User Query
                      │
                      ▼
                  ┌───────┐
                  │Planner│
                  └───┬───┘
                      ▼
                 ┌──────────┐
                 │Retriever │
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │Reasoning │
                 │   LLM    │
                 └────┬─────┘
                      ▼
                  ┌───────┐
                  │ Critic│
                  └───┬───┘
                      │
                ┌─────┴─────┐
                ▼           ▼
              PASS         FAIL
                │           │
                │         Retry
                │           │
                │        Refine
                │           │
                └─────◄─────┘
                      │
                      ▼
                   Finalize
```

### Key design decision

The system separates:

**planning ≠ retrieval ≠ generation ≠ evaluation ≠ refinement**

---

# 🔥 Engineering Highlights

### Evaluation → Refinement

Instead of:

```text id="1n6g7x"
Generate → Return
```

Aetherion uses:

```text id="n9q1j4"
Generate → Evaluate → Refine / Finalize
```

### Multi-LLM Routing

Supports:

* OpenAI
* Anthropic
* Groq
* Hugging Face

### Failure Handling

| Failure          | Response          |
| ---------------- | ----------------- |
| LLM timeout      | Provider fallback |
| Weak generation  | Retry             |
| Retrieval noise  | Context filtering |
| Retry exhaustion | Degraded response |

---

# 🛠️ Tech Stack

**Python · FastAPI · ChromaDB · sentence-transformers · OpenAI · Anthropic · Groq · Hugging Face · Tailwind CSS · Docker · Vercel · Railway**

---

# 🚀 Run Locally

```bash id="d0a7cu"
git clone https://github.com/kcsanjayj/Aetherion.git
cd Aetherion

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python start.py
```

Configure the required API keys through environment variables.

**Never commit secrets to Git.**

---

# 📊 Evaluation

Aetherion is designed to compare:

```text id="4v4t4q"
Baseline RAG
     vs
Aetherion
```

Key metrics:

**Correctness · Groundedness · Retrieval Quality · Citation Quality · Hallucination Rate · Latency · Retry Rate · Cost / Query**

> Performance improvements should only be claimed after reproducible benchmarking.

---

# 🚧 Limitations

* Additional LLM calls increase latency and cost.
* Critics can make incorrect judgments.
* Retrieval quality affects final answers.
* Self-correction can occasionally reinforce errors.

---

## ⭐ Why Aetherion?

Traditional RAG:

```text id="pry6xk"
Retrieve → Generate → Return
```

Aetherion:

```text id="r6f9gq"
Retrieve → Generate → Evaluate → Refine → Finalize
```

> **Aetherion explores what happens when evaluation becomes a first-class part of the RAG generation pipeline.**

---

## 📜 License

MIT License.

**Built by [kcsanjayj](https://github.com/kcsanjayj)**

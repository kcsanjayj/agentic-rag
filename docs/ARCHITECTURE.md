# Architecture Documentation

## System Architecture

### High-Level Flow

```
User Query
    ↓
Orchestrator
    ↓
Retrieval → Reasoning → Critic → Evaluation
    ↓
Final Answer (loop if needed)
```

### Detailed Component Diagram

![System Design](../docs/system-design.png)

## Agent Breakdown

### 1. Orchestrator Agent
- **Role:** Central controller
- **Responsibilities:**
  - Route queries to appropriate agents
  - Manage retry loops (max 3 iterations)
  - Track execution state
  - Compile final response

### 2. Retrieval Agent
- **Role:** Document fetching
- **Method:** Hybrid retrieval (Dense + Sparse)
  - Dense: sentence-transformers embeddings
  - Sparse: BM25 keyword matching
  - Reranking: Combined scoring
- **Output:** Top-k relevant chunks

### 3. Reasoning Agent
- **Role:** Answer synthesis
- **Input:** Retrieved chunks + user query
- **Process:**
  - Analyze context
  - Generate coherent response
  - Include citations
- **Output:** Initial answer draft

### 4. Critic Agent
- **Role:** Quality reviewer
- **Checks:**
  - Missing information
  - Weak reasoning
  - Uncited claims
  - Factual errors
- **Output:** Improvement suggestions

### 5. Evaluation Agent
- **Role:** Quality scorer
- **Metrics:**
  - Relevance (0-10)
  - Grounding (0-10)
  - Completeness (0-10)
- **Threshold:** 7.5/10 for passing
- **Action:** Trigger retry if below threshold

## Data Flow

```
1. Document Upload
   PDF/TXT/DOCX → Text Extraction → Chunking (500 tokens) → Embeddings → ChromaDB

2. Query Processing
   User Query → Embedding → Vector Search + BM25 → Reranking → Top 5 chunks

3. Answer Generation
   Chunks + Query → LLM → Initial Answer → Critic Review → Evaluation

4. Self-Correction (if needed)
   Low Score → More Context → Regenerate → Re-evaluate (max 3x)

5. Response
   Final Answer + Sources + Score + Execution Trace
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Multiple providers (Gemini, OpenAI, NVIDIA, etc.) |
| Frontend | Vanilla JS + CSS |
| Deployment | Docker + Docker Compose |

## Design Decisions

### Why Multi-Agent?
- Separation of concerns enables optimization
- Each agent independently improvable
- Clear failure points for debugging

### Why Hybrid Retrieval?
- Dense captures semantic similarity
- Sparse handles exact keyword matches
- Reranker optimizes both signals

### Why Evaluation Loop?
- Automated quality gate
- Prevents bad outputs reaching users
- Metrics drive improvement

## Scalability Considerations

- **Stateless API:** Easy horizontal scaling
- **Document Isolation:** Namespace-based separation
- **Singleton Embeddings:** Model loaded once, reused
- **Iteration Cap:** Prevents runaway costs

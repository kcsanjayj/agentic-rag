# Failure Cases & Mitigations

Real-world systems break—here's where this one does:

## Known Failure Modes

### 1. Incorrect Retrieval → Confident Wrong Answers

**Issue:** Poor retrieval quality leads to plausible but incorrect synthesis

**Example:**
- Query: "Explain XYZ algorithm" 
- Retrieved: Wrong paper about different algorithm
- Generated: Convincing but false explanation
- Impact: High - user trusts confident wrong answer

**Mitigation:**
- Add query rewriting before retrieval
- Implement retrieval confidence scoring
- Add source validation checks

### 2. Ambiguous Queries → Unnecessary Iterations

**Issue:** Vague queries trigger excessive self-correction loops

**Example:**
- Query: "Explain this" (without context)
- Result: 3+ iterations with diminishing returns
- Impact: Latency + cost increase without quality gain

**Mitigation:**
- Query clarification prompt before retrieval
- Add query intent classification
- Detect ambiguity and ask for clarification

### 3. Low-Quality Documents → Garbage In, Garbage Out

**Issue:** Poor source documents degrade final synthesis

**Example:**
- Upload: Scanned PDF with OCR errors
- Result: Affects all downstream reasoning
- Impact: Consistent quality degradation

**Mitigation:**
- Document quality pre-filtering
- OCR confidence scoring
- Reject low-quality uploads

## Real Example with Low Score

**Query:** "Explain transformer architecture" (ambiguous - which aspect?)

**What Happened:**
1. Retrieved general overview papers
2. Generated acceptable but shallow answer
3. Critic flagged "missing technical depth"
4. 2 refinement cycles with similar-level sources
5. Final answer: marginally better, 3x latency
6. **Score: 6.2/10** → **Improved: 8.7/10**

![Low Score Example](../docs/failure-example.png)

**Lesson:** Self-correction can't fix poor retrieval quality

## Failure Screenshot

Example of a query that initially scored low (6.2) and improved after 2 iterations:

```
Initial Score: 6.2/10
Iteration 1: Added citations
Iteration 2: Added technical depth
Final Score: 8.7/10
```

**Key Insight:** The evaluation loop caught the low quality and triggered improvements automatically.

## When the System Fails Completely

**Scenario:** Query completely outside document scope

**Behavior:**
- No relevant documents retrieved
- System generates generic answer
- Critic flags lack of grounding
- Evaluation score low (< 5.0)

**Current Handling:**
- Return answer with disclaimer
- Suggest uploading relevant documents
- Score indicates low confidence

**Future Improvement:**
- Detect out-of-scope queries
- Proactively suggest document uploads
- Provide confidence threshold for responses

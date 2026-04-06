import requests

# Level 4 Agentic RAG Verification
print("=" * 70)
print("LEVEL 4 AGENTIC RAG - FINAL VERIFICATION")
print("=" * 70)

query = "The composite data is converted to binary classification. Masking removes noise."
response = requests.post('http://localhost:8000/api/v1/query', 
                        json={'query': query, 'top_k': 3})
data = response.json()
answer = data.get('answer', '')

# Check all Level 4 features
features = {
    "PLAN section": "PLAN:" in answer and "Analyze query" in answer,
    "TOOLS USED section": "TOOLS USED:" in answer and "interpret_tool" in answer,
    "ITERATION section": "ITERATION:" in answer and "Attempt" in answer,
    "CONFIDENCE": "CONFIDENCE:" in answer,
    "SYSTEM MODE": "SYSTEM MODE:" in answer,
    "AGENT ACTIONS": "AGENT ACTIONS:" in answer,
    "CRITIC VERDICT": "CRITIC VERDICT:" in answer,
    "SUMMARY": "SUMMARY:" in answer,
    "INSIGHT": "INSIGHT:" in answer,
    "EVALUATION": "EVALUATION:" in answer
}

print("\nLevel 4 Features:")
print("-" * 50)
for feature, present in features.items():
    status = "✅" if present else "❌"
    print(f"{status} {feature}")

all_present = all(features.values())
print("\n" + "=" * 70)
print(f"{'✅ LEVEL 4 AGENTIC RAG COMPLETE' if all_present else '⚠️ SOME FEATURES MISSING'}")
print("=" * 70)

# Show output structure
print("\nOutput Structure:")
print("-" * 70)
lines = answer.split('\n')
for line in lines[:50]:
    if any(x in line for x in ['PLAN:', 'TOOLS USED:', 'CONFIDENCE:', 'SYSTEM MODE:', 
                                'AGENT ACTIONS:', 'ITERATION:', 'CRITIC VERDICT:', 
                                'SUMMARY:', 'INSIGHT:', 'EVALUATION:']):
        print(line)

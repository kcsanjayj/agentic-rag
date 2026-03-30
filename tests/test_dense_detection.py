import requests

# Test 10/10 fixes for technical/dense detection and domain reasoning
print("=" * 70)
print("10/10 TECHNICAL/DENSE DETECTION TEST")
print("=" * 70)

# Test with "range estimates combined conservatively" type content
response = requests.post('http://localhost:8000/api/v1/query', 
                        json={'query': 'What does range estimates combined conservatively mean?', 'top_k': 3})
data = response.json()
answer = data.get('answer', '')

print("\n--- FULL RESPONSE ---")
print(answer[:1200])

# Check for correct behavior
print("\n" + "=" * 70)
print("VERIFICATION CHECKS:")
print("=" * 70)

if "sparse reasoning" in answer.lower() or "Sparse-content reasoning" in answer:
    print("✅ MODE: Sparse reasoning (not fallback)")
else:
    print("❌ MODE: Not using sparse reasoning")

if "conservative aggregation" in answer.lower() or "underestimation of uncertainty" in answer.lower():
    print("✅ INSIGHT: Domain-specific reasoning present")
else:
    print("❌ INSIGHT: Missing domain-specific reasoning")

if "NO (already optimal)" in answer:
    print("✅ REWRITE: Truthful (NO)")
else:
    print("❌ REWRITE: Not showing correct status")

print("\n" + "=" * 70)

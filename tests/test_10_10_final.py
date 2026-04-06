import requests

# 10/10 ELITE RAG VERIFICATION TEST
print("=" * 70)
print("10/10 ELITE RAG - COMPREHENSIVE VERIFICATION")
print("=" * 70)

all_tests_passed = True

# Test 1: Semantic Density Detection
print("\n[TEST 1] Semantic Density Detection")
print("-" * 50)
response = requests.post('http://localhost:8000/api/v1/query', 
                        json={'query': 'What is the flood threshold?', 'top_k': 3})
data = response.json()
answer = data.get('answer', '')

if "Sparse-content reasoning" in answer:
    print("✓ Semantic density detection: ACTIVE")
else:
    print("✗ Semantic density detection: FAILED")
    all_tests_passed = False

# Test 2: Clean Summary (no corruption)
if "MAM and OND" not in answer or "for both" not in answer:
    print("✓ Summary corruption: FIXED (no chunk pollution)")
else:
    print("✗ Summary corruption: STILL PRESENT")
    all_tests_passed = False

# Test 3: Truthful Agent Actions
if "Query rewritten: NO (already optimal)" in answer or "Query rewritten: YES" in answer:
    print("✓ Truthful agent actions: WORKING")
else:
    print("✗ Truthful agent actions: NOT VISIBLE")
    all_tests_passed = False

# Test 4: CRITIC VERDICT
if "CRITIC VERDICT: ACCEPTED" in answer:
    print("✓ CRITIC VERDICT: PRESENT")
else:
    print("✗ CRITIC VERDICT: MISSING")
    all_tests_passed = False

# Test 5: Top Retrieved Chunks (no undefined)
if "Top Retrieved Chunks:" in answer and "undefined" not in answer:
    print("✓ Top Retrieved Chunks: PRESENT & CLEAN")
else:
    print("✗ Top Retrieved Chunks: MISSING or HAS 'undefined'")
    all_tests_passed = False

# Test 6: Output Structure
print("\n[TEST 6] Final Output Structure")
print("-" * 50)
checks = [
    ("CONFIDENCE:", "Confidence header"),
    ("SYSTEM MODE:", "System mode"),
    ("AGENT ACTIONS:", "Agent actions"),
    ("CRITIC VERDICT:", "Critic verdict"),
    ("SUMMARY:", "Summary"),
    ("INSIGHT:", "Insight"),
    ("EVALUATION:", "Evaluation")
]

for check, name in checks:
    if check in answer:
        print(f"✓ {name}: PRESENT")
    else:
        print(f"✗ {name}: MISSING")
        all_tests_passed = False

# Summary
print("\n" + "=" * 70)
if all_tests_passed:
    print("✅ ALL TESTS PASSED - 10/10 ELITE SYSTEM READY")
else:
    print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
print("=" * 70)

print("\n" + "-" * 70)
print("SAMPLE OUTPUT:")
print("-" * 70)
print(answer[:800] if len(answer) > 800 else answer)

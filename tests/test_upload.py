#!/usr/bin/env python3
"""Test upload endpoint"""

import requests
import os

# Create test file
test_content = "This is a test document for upload testing. It has multiple sentences."
with open("test_upload.txt", "w") as f:
    f.write(test_content)

# Test upload
url = "http://localhost:8080/api/v1/upload"
headers = {
    "X-API-Key": "test-secret-12345678901234567890",
    "X-User-Api-Key": "sk-dummy-test-key-12345"
}

files = {"file": ("test_upload.txt", open("test_upload.txt", "rb"), "text/plain")}

print("Testing upload endpoint...")
try:
    response = requests.post(url, headers=headers, files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
finally:
    files["file"][1].close()

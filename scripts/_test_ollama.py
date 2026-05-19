import requests, json, time
print("Testing Ollama qwen3:4b...")
t0 = time.time()
payload = {
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": 'Return ONLY this JSON, nothing else:\n{"winner":"A","confidence":0.9,"reason":"A has correct boundaries"}'}],
    "stream": False,
    "think": False,
    "options": {"num_predict": 500}
}
r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600, proxies={"http": None, "https": None})
elapsed = round(time.time() - t0, 1)
result = r.json()
content = result.get("message", {}).get("content", "")
print(f"Done in {elapsed}s")
print("RAW:", repr(content[:400]))
# Try to parse JSON from response
import re
m = re.search(r'\{.*\}', content, re.DOTALL)
if m:
    try:
        parsed = json.loads(m.group())
        print("PARSED JSON:", json.dumps(parsed, indent=2))
    except:
        print("Could not parse JSON")

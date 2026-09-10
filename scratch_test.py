import urllib.request
import re

url = "https://www.moneycontrol.com/commodity/goldm-price.html"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
with urllib.request.urlopen(req, timeout=10) as r:
    html = r.read().decode("utf-8", errors="ignore")

print("Has 'advanced':", "advanced" in html.lower())
print("Has 'chart':", "chart" in html.lower())
for line in html.splitlines():
    if "advanced" in line.lower() or "chart" in line.lower():
        if len(line.strip()) < 150:
            print("LINE:", line.strip())

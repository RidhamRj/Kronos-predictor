import requests
import json
import time
import re

def get_mc():
    v = str(time.time())
    html = requests.get(f'https://www.moneycontrol.com/commodity/mcx-goldm-price/?type=futures&exp=2026-10-05&v={v}', headers={'User-Agent': 'Mozilla/5.0'}).text
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html)
    return json.loads(m.group(1))['props']['pageProps']['data']['commodityData']['lastupdTime']

print(get_mc())

import httpx, xml.etree.ElementTree as ET
feeds = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmag": "https://bitcoinmagazine.com/feed",
}
for name, url in feeds.items():
    try:
        r = httpx.get(url, timeout=12, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        print(f"{name}: HTTP {r.status_code}, {len(items)} items")
        if items:
            t = items[0].findtext("title"); d = items[0].findtext("pubDate")
            print(f"   latest: [{d}] {t[:80] if t else ''}")
    except Exception as e:
        print(f"{name}: FAIL {str(e)[:100]}")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from config import TMDB_API_KEY, TMDB_BASE_URL

def check(label, url, params):
    resp = requests.get(url, params={**params, "api_key": TMDB_API_KEY})
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("results", [data])
        print(f"  [PASS] {label} — got {len(results)} result(s)")
        if results:
            first = results[0]
            print(f"         First result: {first.get('title') or first.get('name') or first.get('original_title')}")
        return True
    else:
        print(f"  [FAIL] {label} — HTTP {resp.status_code}: {resp.text[:120]}")
        return False

if not TMDB_API_KEY:
    print("[ERROR] TMDB_API_KEY not set in .env")
    sys.exit(1)

print("Verifying TMDB API...\n")

results = [
    check("Search movie (Hollywood)",  f"{TMDB_BASE_URL}/search/movie",  {"query": "Inception"}),
    check("Search movie (Indian)",     f"{TMDB_BASE_URL}/search/movie",  {"query": "Baahubali"}),
    check("Search person",             f"{TMDB_BASE_URL}/search/person", {"query": "Shah Rukh Khan"}),
    check("Similar movies",            f"{TMDB_BASE_URL}/movie/27205/similar", {}),
    check("Movie keywords",            f"{TMDB_BASE_URL}/movie/27205/keywords", {}),
]

print()
if all(results):
    print("All TMDB checks passed. Ready to build.")
else:
    print("Some checks failed. Fix the API key or network before proceeding.")

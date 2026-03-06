import requests

SERPAPI_KEY = ""

def search_web(claim, num_results=3):
    url = "https://serpapi.com/search.json"
    
    params = {
        "engine": "google",
        "q": claim,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    results = []
    
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })
    
    return results
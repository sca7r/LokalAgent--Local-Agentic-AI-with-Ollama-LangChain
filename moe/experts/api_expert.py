"""
moe/experts/api_expert.py  
API expert — HTTP calls, no LLM.
"""
import json, requests
from moe.expert_contract import ok, err


def run(instruction: str, context: dict = None) -> dict:
    try:
        params = json.loads(instruction)
    except json.JSONDecodeError:
        import re
        url = re.search(r'https?://\S+', instruction)
        if url:
            params = {"url": url.group(0), "method": "GET"}
        else:
            return err("Provide JSON with url/method or a URL")
    try:
        res = requests.request(
            method=params.get("method", "GET"),
            url=params["url"],
            headers=params.get("headers", {}),
            json=params.get("body"),
            timeout=15,
        )
        try:
            data = res.json()
            return ok(value=data, unit="json",
                      summary=f"API call to {params['url']} returned {res.status_code}",
                      raw=json.dumps(data)[:1000])
        except Exception:
            return ok(value=res.text[:500], unit="text",
                      summary=f"API call returned status {res.status_code}",
                      raw=res.text[:500])
    except Exception as e:
        return err(str(e))

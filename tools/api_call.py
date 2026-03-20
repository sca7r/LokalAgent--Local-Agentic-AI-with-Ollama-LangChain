from langchain.tools import Tool
import requests
import json


def _make_api_call(input_str: str) -> str:
    """
    Make an HTTP API call.
    Input format (JSON string):
    {
        "url": "https://api.example.com/endpoint",
        "method": "GET",                          # GET, POST, PUT, DELETE
        "headers": {"Authorization": "Bearer ..."},  # optional
        "body": {"key": "value"}                  # optional, for POST/PUT
    }
    """
    try:
        params = json.loads(input_str)
    except json.JSONDecodeError:
        return "Error: Input must be a valid JSON string with keys: url, method, headers (optional), body (optional)."

    url = params.get("url")
    method = params.get("method", "GET").upper()
    headers = params.get("headers", {})
    body = params.get("body", None)

    if not url:
        return "Error: 'url' is required."

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=body,
            timeout=15,
        )
        try:
            result = response.json()
            return json.dumps(result, indent=2)
        except Exception:
            return response.text
    except requests.exceptions.RequestException as e:
        return f"API call failed: {e}"


def get_api_tool():
    """Returns a generic HTTP API call tool."""
    return Tool(
        name="api_call",
        func=_make_api_call,
        description=(
            "Make an HTTP API call to any external service. "
            "Input must be a JSON string with keys: "
            "'url' (required), 'method' (GET/POST/PUT/DELETE, default GET), "
            "'headers' (optional dict), 'body' (optional dict for POST/PUT). "
            'Example: {"url": "https://api.github.com/users/octocat", "method": "GET"}'
        ),
    )
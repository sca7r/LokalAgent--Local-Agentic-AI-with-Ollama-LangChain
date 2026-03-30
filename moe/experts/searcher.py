"""
moe/experts/searcher.py

WHY searcher returns raw text always:
The searcher does not know what the downstream expert needs.
Trying to extract "the right value" here is fragile and task-specific.
Instead we return the full search text — downstream experts extract
what they need using their own context (the original task).

This is how production pipelines work:
  - Retrieval step: get relevant text
  - Processing step: extract/transform what's needed from that text
"""

import os, re
from dotenv import load_dotenv

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_project_root, ".env"))


def run(instruction: str, context: dict = None, model: str = "deepseek-r1:7b") -> dict:
    from moe.expert_contract import ok, err

    key = os.getenv("TAVILY_API_KEY", "")
    if not key or key.startswith("tvly-..."):
        return err("TAVILY_API_KEY not set", expert="web_search", task=instruction)

    try:
        from tavily import TavilyClient
        results = TavilyClient(api_key=key).search(instruction, max_results=3)
        raw = "\n\n".join(
            f"Source: {r['url']}\n{r['content'][:500]}"
            for r in results.get("results", [])
        )
        if not raw:
            return err("No results returned", expert="web_search", task=instruction)

        # Always return raw text — downstream experts extract what they need
        # WHY: searcher doesn't know if next step needs a number, a list,
        # or a paragraph. Returning raw text preserves all information.
        return ok(
            value      = raw,
            value_type = "text",
            unit       = "",
            summary    = f"Search results for: {instruction}",
            expert     = "web_search",
            task       = instruction,
            raw        = raw[:500],
            confidence = "high",
        )
    except Exception as e:
        return err(str(e), expert="web_search", task=instruction)

from dotenv import load_dotenv
import os

load_dotenv()


def get_search_tool():
    """
    Returns a Tavily web search tool if TAVILY_API_KEY is set,
    otherwise returns None so the agent starts cleanly without it.
    Sign up free at https://tavily.com to enable web search.
    """
    key = os.getenv("TAVILY_API_KEY", "")
    if not key or key.startswith("tvly-..."):
        print("[INFO] TAVILY_API_KEY not set — web search disabled.")
        return None

    from langchain_community.tools.tavily_search import TavilySearchResults
    return TavilySearchResults(
        max_results=5,
        description=(
            "Search the web for current information. "
            "Use this when you need up-to-date facts, recent news, "
            "or any information you don't already know."
        ),
    )
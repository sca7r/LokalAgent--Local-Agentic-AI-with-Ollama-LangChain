"""
tools/doc_search.py
LangChain tool wrapping LokalAgent's PageIndex-style retriever.
"""

from langchain.tools import Tool

_current_index = {"data": None, "filename": None}


def set_index(index: dict):
    _current_index["data"] = index
    _current_index["filename"] = index.get("filename", "document")


def clear_index():
    _current_index["data"] = None
    _current_index["filename"] = None


def _search_document(query: str) -> str:
    if _current_index["data"] is None:
        return "No document has been uploaded yet. Please upload a PDF first."

    from rag.retriever import retrieve
    index = _current_index["data"]
    model = index.get("model_used", "llama3:latest")
    return retrieve(query=query, index=index, model_name=model)


def get_doc_search_tool() -> Tool:
    return Tool(
        name="document_search",
        func=_search_document,
        description=(
            "Search the uploaded PDF document for information. "
            "Use this whenever the user asks about the content of an uploaded document or file. "
            "Input: a natural language question about the document."
        ),
    )
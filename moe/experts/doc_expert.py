"""
moe/experts/doc_expert.py
Document expert — PageIndex RAG.
"""
from moe.expert_contract import ok, err


def run(instruction: str, context: dict = None) -> dict:
    try:
        from tools.doc_search import _search_document
        result = _search_document(instruction)
        return ok(value=result, unit="text",
                  summary=f"Document search: {instruction[:60]}",
                  raw=result)
    except Exception as e:
        return err(str(e))

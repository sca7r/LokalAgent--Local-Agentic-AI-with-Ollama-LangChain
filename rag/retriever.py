"""
rag/retriever.py
PageIndex-style iterative retrieval — uses reasoning over the tree index,
not vector similarity. Mirrors the 5-step loop from the PageIndex paper:
  1. Read TOC
  2. Select most relevant section
  3. Extract information
  4. Check if sufficient → if not, loop
  5. Return gathered context
"""

import json
import asyncio
from rag.utils import (
    llm_completion, llm_acompletion, extract_json,
    structure_to_list, get_text_of_pdf_pages, SimpleLogger,
)


def build_toc_string(structure: list, indent: int = 0) -> str:
    """Build a human-readable TOC string from the tree for the LLM to reason over."""
    lines = []
    for node in structure:
        prefix = "  " * indent
        node_id = node.get('node_id', '?')
        title   = node.get('title', 'Untitled')
        start   = node.get('start_index', '?')
        end     = node.get('end_index', '?')
        summary = node.get('summary', '')

        line = f"{prefix}[{node_id}] {title} (pages {start}-{end})"
        if summary:
            line += f"\n{prefix}    → {summary[:120]}"
        lines.append(line)

        if node.get('nodes'):
            lines.append(build_toc_string(node['nodes'], indent + 1))

    return "\n".join(lines)


def select_nodes_to_explore(query: str, toc_string: str, already_explored: list,
                             model: str, doc_name: str) -> dict:
    """
    Step 2: Ask LLM which node_ids to explore next given the query and TOC.
    Returns {"thinking": "...", "node_ids": [...], "sufficient": "yes/no"}
    """
    explored_str = ", ".join(already_explored) if already_explored else "none"

    prompt = f"""You are a document retrieval assistant for "{doc_name}".

Your task: Given a user query and the document's table of contents,
decide which sections to read next to answer the query.

User query: {query}

Already explored sections: {explored_str}

Table of contents:
{toc_string}

Return JSON only:
{{
    "thinking": "<reason about which sections are most likely to contain the answer>",
    "node_ids": ["<node_id_1>", "<node_id_2>"],
    "sufficient": "<yes if already explored sections are enough to answer, no otherwise>"
}}

Rules:
- Pick 1-2 most relevant unexplored node_ids
- If already_explored sections clearly contain the answer, set sufficient = "yes" and node_ids = []
- If no sections seem relevant, pick the most likely ones
Directly return the JSON. Do not output anything else."""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    if not isinstance(result, dict):
        return {"thinking": "", "node_ids": [], "sufficient": "no"}
    return result


def get_node_by_id(structure: list, node_id: str) -> dict | None:
    """Find a node in the tree by its node_id."""
    all_nodes = structure_to_list(structure)
    for node in all_nodes:
        if node.get('node_id') == node_id:
            return node
    return None


def get_node_text(node: dict, page_list: list = None) -> str:
    """Get the text content of a node — from stored text or from page_list."""
    if node.get('text'):
        return node['text']
    if page_list and node.get('start_index') and node.get('end_index'):
        return get_text_of_pdf_pages(page_list, node['start_index'], node['end_index'])
    return f"[Section: {node.get('title', 'Unknown')} — text not available]"


def generate_final_answer(query: str, gathered_context: list, model: str, doc_name: str) -> str:
    """
    Step 5: Given gathered context from multiple sections, generate the final answer.
    """
    context_str = "\n\n---\n\n".join([
        f"[Section: {c['title']} | Pages {c['start']}-{c['end']}]\n{c['text'][:2000]}"
        for c in gathered_context
    ])

    prompt = f"""You are answering a question about the document "{doc_name}".

Based on the retrieved sections below, answer the user's query.
Be specific and cite which section/pages the information comes from.

User query: {query}

Retrieved sections:
{context_str}

Provide a clear, complete answer based on the retrieved content."""

    response, _ = llm_completion(model, prompt)
    return response


def retrieve(query: str, index: dict, model_name: str = "llama3:latest",
             max_iterations: int = 3) -> str:
    """
    Main retrieval function — implements the PageIndex 5-step iterative loop.

    1. Build TOC string from index
    2. Ask LLM which sections to explore
    3. Fetch those sections' text
    4. Check if sufficient
    5. Loop until sufficient or max_iterations reached
    6. Generate final answer from gathered context
    """
    logger = SimpleLogger("Retriever")
    structure = index.get('structure', [])
    doc_name  = index.get('doc_name', index.get('filename', 'document'))
    model     = model_name

    if not structure:
        return "No document index available. Please upload and index a PDF first."

    toc_string     = build_toc_string(structure)
    gathered       = []   # list of {"title", "start", "end", "text"}
    explored_ids   = []

    logger.info(f"Retrieving for query: '{query}'")

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")

        # Step 2: Select nodes to explore
        selection = select_nodes_to_explore(
            query, toc_string, explored_ids, model, doc_name
        )
        logger.info(f"Thinking: {selection.get('thinking', '')[:100]}")
        logger.info(f"Selected: {selection.get('node_ids', [])}")

        # Step 4: Check if we already have enough
        if selection.get('sufficient') == 'yes':
            logger.info("LLM says sufficient context gathered")
            break

        node_ids = selection.get('node_ids', [])
        if not node_ids:
            logger.info("No new nodes to explore")
            break

        # Step 3: Fetch text from selected nodes
        found_new = False
        for node_id in node_ids:
            if node_id in explored_ids:
                continue

            node = get_node_by_id(structure, node_id)
            if node is None:
                logger.info(f"Node {node_id} not found in index")
                continue

            text = get_node_text(node)
            gathered.append({
                'title': node.get('title', 'Unknown'),
                'start': node.get('start_index', '?'),
                'end':   node.get('end_index', '?'),
                'text':  text,
            })
            explored_ids.append(node_id)
            found_new = True
            logger.info(f"Fetched: [{node_id}] {node.get('title')} (pages {node.get('start_index')}-{node.get('end_index')})")

        if not found_new:
            logger.info("No new content found — stopping")
            break

    if not gathered:
        return f"Could not find relevant information in '{doc_name}' for your query."

    # Step 5: Generate final answer
    logger.info(f"Generating answer from {len(gathered)} sections")
    answer = generate_final_answer(query, gathered, model, doc_name)

    # Append source references
    sources = "\n".join([
        f"  • {c['title']} (pages {c['start']}–{c['end']})"
        for c in gathered
    ])
    return f"{answer}\n\n**Sources consulted:**\n{sources}"
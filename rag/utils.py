"""
rag/utils.py
Utility functions for LokalAgent's PageIndex-style RAG.
Adapted from VectifyAI/PageIndex — replaces litellm with langchain_ollama,
pymupdf with pypdf, and litellm tokenizer with char-based approximation.
"""

import json
import re
import asyncio
import logging
import copy
from pypdf import PdfReader
from langchain_ollama import ChatOllama

# ── LLM helpers ──────────────────────────────────────────────────────────────

def get_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        temperature=0,
    )


def llm_completion(model: str, prompt: str, chat_history: list = None) -> tuple[str, str]:
    """
    Sync LLM call. Returns (content, finish_reason).
    finish_reason is 'finished' or 'max_output_reached'.
    """
    llm = get_llm(model)
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Build message objects
            from langchain_core.messages import HumanMessage, AIMessage
            lc_messages = []
            for m in messages:
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    lc_messages.append(AIMessage(content=m["content"]))
            response = llm.invoke(lc_messages)
            content = response.content
            # Ollama doesn't expose finish_reason reliably — treat all as finished
            return content, "finished"
        except Exception as e:
            logging.error(f"llm_completion attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                import time; time.sleep(2)
    return "", "error"


async def llm_acompletion(model: str, prompt: str) -> str:
    """Async LLM call — runs sync call in thread pool to avoid blocking."""
    loop = asyncio.get_event_loop()
    content, _ = await loop.run_in_executor(None, llm_completion, model, prompt, None)
    return content


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(content: str) -> dict | list:
    """Robustly extract JSON from LLM response — handles phi3/llama3 verbose output."""
    if not content:
        return {}

    original = content

    # 1. Try ```json ... ``` block
    start = content.find("```json")
    if start != -1:
        start += 7
        end = content.rfind("```")
        if end > start:
            content = content[start:end].strip()

    # 2. Try to find the first { or [ and last } or ]
    # This handles cases where the model adds text before/after JSON
    def try_parse(s):
        s = s.replace("None", "null").replace("True", "true").replace("False", "false")
        s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
        return json.loads(s)

    # Try full content first
    try:
        return try_parse(content.strip())
    except Exception:
        pass

    # Try extracting just the JSON object/array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = content.find(start_char)
        if start_idx == -1:
            continue
        # Walk backward from end to find matching close
        depth = 0
        end_idx = -1
        in_string = False
        escape = False
        for i, ch in enumerate(content[start_idx:], start_idx):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx != -1:
            candidate = content[start_idx:end_idx+1]
            try:
                return try_parse(candidate)
            except Exception:
                # Try cleaning up truncated JSON
                try:
                    # Add missing closing brackets
                    fixed = candidate
                    open_braces = fixed.count('{') - fixed.count('}')
                    open_brackets = fixed.count('[') - fixed.count(']')
                    fixed += '}' * open_braces + ']' * open_brackets
                    return try_parse(fixed)
                except Exception:
                    pass

    logging.error(f"Failed to parse JSON: {original[:200]}")
    return {}


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = None) -> int:
    """Approximate token count — 1 token ≈ 4 chars."""
    if not text:
        return 0
    return len(text) // 4


# ── PDF page extraction ───────────────────────────────────────────────────────

def get_page_tokens(pdf_path: str, model: str = None) -> list[tuple[str, int]]:
    """
    Extract pages from PDF.
    Returns list of (page_text, token_count) tuples — same format as original PageIndex.
    """
    reader = PdfReader(pdf_path)
    page_list = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        tokens = count_tokens(text, model)
        page_list.append((text, tokens))
    return page_list


def get_pdf_name(pdf_path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(pdf_path))[0]


def get_text_of_pdf_pages(page_list: list, start_page: int, end_page: int) -> str:
    """Get combined text for pages start_page..end_page (1-indexed)."""
    texts = []
    for i in range(start_page - 1, min(end_page, len(page_list))):
        texts.append(page_list[i][0])
    return "\n\n".join(texts)


# ── Physical index tagging ────────────────────────────────────────────────────

def tag_pages(page_list: list, start_index: int = 1) -> list[str]:
    """
    Wrap each page in <physical_index_X> tags — the core PageIndex innovation.
    Returns list of tagged page strings.
    """
    tagged = []
    for i, (text, _) in enumerate(page_list):
        idx = start_index + i
        tagged.append(f"<physical_index_{idx}>\n{text}\n<physical_index_{idx}>\n\n")
    return tagged


def convert_physical_index_to_int(data):
    """Convert '<physical_index_X>' strings to integers."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'physical_index' in item:
                val = item['physical_index']
                if isinstance(val, str):
                    match = re.search(r'physical_index_(\d+)', val)
                    if match:
                        item['physical_index'] = int(match.group(1))
                    else:
                        item['physical_index'] = None
    elif isinstance(data, str):
        match = re.search(r'physical_index_(\d+)', data)
        if match:
            return int(match.group(1))
        return None
    return data


# ── Page grouping ─────────────────────────────────────────────────────────────

def page_list_to_group_text(
    page_contents: list[str],
    token_lengths: list[int],
    max_tokens: int = 8000,   # lower than original 20000 — Ollama has smaller context
    overlap_page: int = 1,
) -> list[str]:
    """
    Split tagged pages into groups that fit within max_tokens.
    Uses overlap_page to preserve context across group boundaries.
    """
    num_tokens = sum(token_lengths)
    if num_tokens <= max_tokens:
        return ["".join(page_contents)]

    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts = max(1, round(num_tokens / max_tokens))
    avg_tokens = (num_tokens // expected_parts + max_tokens) // 2

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > avg_tokens and current_subset:
            subsets.append("".join(current_subset))
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        current_subset.append(page_content)
        current_token_count += page_tokens

    if current_subset:
        subsets.append("".join(current_subset))

    print(f"[PageIndex] Divided into {len(subsets)} groups")
    return subsets


# ── Tree structure helpers ────────────────────────────────────────────────────

def structure_to_list(structure) -> list:
    """Flatten nested tree to a list of all nodes."""
    if isinstance(structure, dict):
        nodes = [structure]
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def write_node_id(data, node_id: int = 0) -> int:
    """Assign sequential node_ids to all nodes in the tree."""
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        if 'nodes' in data:
            node_id = write_node_id(data['nodes'], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = write_node_id(item, node_id)
    return node_id


def add_node_text(node, page_list: list):
    """Add raw text to each node from page_list using start_index/end_index."""
    if isinstance(node, dict):
        start = node.get('start_index')
        end   = node.get('end_index')
        if start and end:
            node['text'] = get_text_of_pdf_pages(page_list, start, end)
        if 'nodes' in node:
            add_node_text(node['nodes'], page_list)
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, page_list)


def remove_structure_text(data):
    """Remove text field from all nodes (after summary generation)."""
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def post_processing(flat_toc: list, total_pages: int) -> list:
    """
    Convert flat list with 'structure' numeric indices (1, 1.1, 1.2, 2, ...)
    into a nested tree, and compute start_index / end_index for each node.
    Mirrors PageIndex's post_processing function.
    """
    if not flat_toc:
        return []

    # Compute start_index for each item
    for item in flat_toc:
        item['start_index'] = item.get('physical_index', 1)

    # Compute end_index = next sibling's start_index - 1, last = total_pages
    for i, item in enumerate(flat_toc):
        if i + 1 < len(flat_toc):
            item['end_index'] = flat_toc[i + 1]['start_index']
        else:
            item['end_index'] = total_pages

    # Build nested tree from structure indices
    root = []
    stack = []  # (node, depth)

    for item in flat_toc:
        structure = item.get('structure', '1')
        depth = len(str(structure).split('.')) if structure else 1

        node = {
            'title':       item.get('title', ''),
            'start_index': item['start_index'],
            'end_index':   item['end_index'],
            'nodes':       [],
        }

        # Pop stack until we find a parent shallower than current depth
        while stack and stack[-1][1] >= depth:
            stack.pop()

        if not stack:
            root.append(node)
        else:
            stack[-1][0]['nodes'].append(node)

        stack.append((node, depth))

    # Clean up empty nodes lists
    def clean(nodes):
        for n in nodes:
            if not n['nodes']:
                del n['nodes']
            else:
                clean(n['nodes'])
    clean(root)

    return root


def add_preface_if_needed(toc: list) -> list:
    """If first section doesn't start on page 1, add a Preface node."""
    if not toc:
        return toc
    first = toc[0]
    if first.get('physical_index', 1) > 1:
        preface = {
            'structure': '0',
            'title': 'Preface',
            'physical_index': 1,
        }
        toc.insert(0, preface)
    return toc


# ── Logger ────────────────────────────────────────────────────────────────────

class SimpleLogger:
    """Simple logger that prints to console with a prefix."""
    def __init__(self, name: str = "PageIndex"):
        self.name = name

    def info(self, msg):
        print(f"[{self.name}] {msg}")

    def error(self, msg):
        print(f"[{self.name}][ERROR] {msg}")

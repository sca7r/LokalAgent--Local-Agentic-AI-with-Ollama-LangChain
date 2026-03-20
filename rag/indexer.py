"""
rag/indexer.py

Implements all 3 processing paths for TOC detection + tree building, plus verification, self-correction, and large node recursion:
  Path A: PDF has TOC with page numbers
  Path B: PDF has TOC but no page numbers
  Path C: No TOC — generate structure from scratch

Plus: verification, self-correction, large node recursion.
"""

import json
import copy
import asyncio
import os
from rag.utils import (
    llm_completion, llm_acompletion, extract_json,
    count_tokens, get_page_tokens, get_pdf_name,
    tag_pages, convert_physical_index_to_int,
    page_list_to_group_text, post_processing,
    add_preface_if_needed, add_node_text, remove_structure_text,
    write_node_id, structure_to_list, SimpleLogger,
)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'toc_check_page_num':     5,   # only check first 5 pages for TOC 
    'max_page_num_each_node': 8,
    'max_token_num_each_node': 6000,
    'if_add_node_id':      'yes',
    'if_add_node_summary': 'no',   # saves many LLM calls
    'if_add_node_text':    'yes',  # keep text so retriever can read it directly
}

# Docs under this threshold skip TOC detection → straight to Path C 
SMALL_DOC_PAGE_THRESHOLD = 15


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TOC DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def toc_detector_single_page(content: str, model: str) -> str:
    """Ask LLM if this page contains a Table of Contents."""
    prompt = f"""Does the following text contain a Table of Contents?
A Table of Contents lists sections/chapters with their titles.
Note: abstract, summary, notation list, figure list are NOT a Table of Contents.

Text: {content[:1500]}

Reply with ONLY this JSON, nothing else:
{{"toc_detected": "yes"}}
or
{{"toc_detected": "no"}}"""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    return result.get('toc_detected', 'no')


def detect_page_index(toc_content: str, model: str) -> str:
    """Check if the TOC contains page numbers."""
    prompt = f"""Does this Table of Contents contain page numbers?

Text: {toc_content[:1000]}

Reply with ONLY this JSON, nothing else:
{{"page_index_given_in_toc": "yes"}}
or
{{"page_index_given_in_toc": "no"}}"""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    return result.get('page_index_given_in_toc', 'no')


def find_toc_pages(page_list: list, opt: dict, logger: SimpleLogger) -> list:
    """Scan first toc_check_page_num pages to find TOC pages."""
    toc_page_list = []
    last_was_yes = False

    for i in range(len(page_list)):
        if i >= opt['toc_check_page_num'] and not last_was_yes:
            break
        result = toc_detector_single_page(page_list[i][0], opt['model'])
        if result == 'yes':
            toc_page_list.append(i)
            last_was_yes = True
        elif result == 'no' and last_was_yes:
            break

    return toc_page_list


def toc_extractor(page_list: list, toc_page_list: list, model: str) -> dict:
    """Extract raw TOC text and detect if page numbers are present."""
    toc_content = ""
    import re
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    # Replace dot leaders with colon
    toc_content = re.sub(r'\.{5,}', ': ', toc_content)
    toc_content = re.sub(r'(?:\. ){5,}\.?', ': ', toc_content)

    has_page_index = detect_page_index(toc_content, model)
    return {
        'toc_content': toc_content,
        'page_index_given_in_toc': has_page_index,
    }


def check_toc(page_list: list, opt: dict, logger: SimpleLogger) -> dict:
    """Top-level TOC check — returns toc_content, toc_page_list, page_index_given_in_toc."""
    toc_page_list = find_toc_pages(page_list, opt, logger)

    if not toc_page_list:
        logger.info("No TOC found")
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}

    logger.info(f"TOC found on pages: {toc_page_list}")
    toc_json = toc_extractor(page_list, toc_page_list, opt['model'])

    if toc_json['page_index_given_in_toc'] == 'yes':
        logger.info("TOC has page numbers")
        return {
            'toc_content': toc_json['toc_content'],
            'toc_page_list': toc_page_list,
            'page_index_given_in_toc': 'yes',
        }

    logger.info("TOC found but no page numbers")
    return {
        'toc_content': toc_json['toc_content'],
        'toc_page_list': toc_page_list,
        'page_index_given_in_toc': 'no',
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — THREE PROCESSING PATHS
# ══════════════════════════════════════════════════════════════════════════════

def toc_transformer(toc_content: str, model: str) -> list:
    """
    Path A/B Step 1: Convert raw TOC text → structured JSON list with
    structure indices (1, 1.1, ...) and page numbers.
    """
    prompt = f"""You are given a table of contents. Transform it into JSON format.

The "structure" field uses numeric hierarchy: first section = "1", 
first subsection = "1.1", second subsection = "1.2", nested = "1.1.1", etc.

Return JSON only:
{{
    "table_of_contents": [
        {{
            "structure": "<x.x.x>",
            "title": "<section title>",
            "page": <page number or null>
        }},
        ...
    ]
}}

Table of contents:
{toc_content}

Directly return the JSON. Do not output anything else."""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    if isinstance(result, dict) and 'table_of_contents' in result:
        return result['table_of_contents']
    if isinstance(result, list):
        return result
    return []


def toc_index_extractor(toc: list, content: str, model: str) -> list:
    """
    Path A Step 2: Given TOC structure + tagged pages,
    find the physical_index for each section.
    """
    prompt = f"""You are given a table of contents in JSON and several tagged pages of a document.

The pages contain tags like <physical_index_X> to indicate page X location.

Your job: add the physical_index to each TOC entry where that section starts.
The "structure" field uses numeric hierarchy (1, 1.1, 1.2, etc.)

Return JSON only — a list:
[
    {{
        "structure": "<x.x.x>",
        "title": "<title>",
        "physical_index": "<physical_index_X>"
    }},
    ...
]

Only add physical_index for sections found in the provided pages.
Directly return the JSON. Do not output anything else.

Table of contents:
{json.dumps(toc, indent=2)}

Document pages:
{content}"""

    response, _ = llm_completion(model, prompt)
    return extract_json(response) or []


def calculate_page_offset(pairs: list) -> int:
    """Calculate the most common offset between PDF page numbers and physical indices."""
    diffs = []
    for p in pairs:
        try:
            diff = p['physical_index'] - p['page']
            diffs.append(diff)
        except (KeyError, TypeError):
            continue
    if not diffs:
        return 0
    counts = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def extract_matching_page_pairs(toc_page: list, toc_physical: list, start_page_index: int) -> list:
    pairs = []
    for phy in toc_physical:
        for pg in toc_page:
            if phy.get('title') == pg.get('title'):
                pi = phy.get('physical_index')
                if pi is not None and int(pi) >= start_page_index:
                    pairs.append({
                        'title': phy['title'],
                        'page': pg.get('page'),
                        'physical_index': pi,
                    })
    return pairs


def add_page_number_to_toc(part: str, structure: list, model: str) -> list:
    """
    Path B: Given tagged pages and TOC structure,
    find where each section physically starts.
    """
    prompt = f"""You are given a JSON structure of a document and a partial part of the document.
Your task: check if each section in the structure starts in this partial document.

The text contains tags like <physical_index_X> to indicate page X.

If a section starts here: set "physical_index": "<physical_index_X>"
If not: set "physical_index": null

Return JSON only — same structure with physical_index filled in:
[
    {{
        "structure": "<x.x.x>",
        "title": "<title>",
        "physical_index": "<physical_index_X> or null"
    }},
    ...
]

Do not change previously filled physical_index values.
Directly return the JSON. Do not output anything else.

Current document part:
{part}

Structure to fill:
{json.dumps(structure, indent=2)}"""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    if isinstance(result, list):
        return result
    return structure


def generate_toc_init(part: str, model: str) -> list:
    """
    Path C Step 1: Generate initial TOC structure from first group of tagged pages.
    """
    prompt = f"""You are an expert in extracting hierarchical tree structures.
Your task: generate the tree structure of the document from the given text.

The "structure" field uses numeric hierarchy:
- First section = "1"
- First subsection = "1.1", second = "1.2"
- Nested = "1.1.1", etc.

The text contains tags like <physical_index_X> to indicate the start/end of page X.
For physical_index: extract the page number where the section STARTS. Keep format "<physical_index_X>".

Return JSON only:
[
    {{
        "structure": "<x.x.x>",
        "title": "<exact title from text, only fix spacing>",
        "physical_index": "<physical_index_X>"
    }},
    ...
]

Directly return the JSON. Do not output anything else.

Document text:
{part}"""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    return result if isinstance(result, list) else []


def generate_toc_continue(toc_content: list, part: str, model: str) -> list:
    """
    Path C Step N: Continue building TOC from next group of tagged pages,
    given the already-built tree as context.
    """
    prompt = f"""You are an expert in extracting hierarchical tree structures.
You are given the tree structure built so far and the next part of the document.
Your task: continue the tree structure to include any NEW sections in the current part.

The "structure" field uses numeric hierarchy (1, 1.1, 1.2, etc.)
Continue from where the previous structure left off.

The text contains tags like <physical_index_X> to indicate page locations.
Keep physical_index in "<physical_index_X>" format.

Return ONLY the NEW entries (not the previous ones) as JSON:
[
    {{
        "structure": "<x.x.x>",
        "title": "<exact title>",
        "physical_index": "<physical_index_X>"
    }},
    ...
]

Directly return the JSON. Do not output anything else.

Previous tree structure:
{json.dumps(toc_content, indent=2)}

Current document part:
{part}"""

    response, _ = llm_completion(model, prompt)
    result = extract_json(response)
    return result if isinstance(result, list) else []


def process_no_toc(page_list: list, start_index: int, opt: dict, logger: SimpleLogger) -> list:
    """Path C: No TOC — generate structure from scratch using tagged pages."""
    logger.info("Path C: Generating TOC from scratch")

    tagged_pages = tag_pages(page_list, start_index)
    token_lengths = [count_tokens(p) for p in tagged_pages]
    groups = page_list_to_group_text(tagged_pages, token_lengths, max_tokens=opt.get('max_token_num_each_node', 6000))

    logger.info(f"Processing {len(groups)} groups")
    toc = generate_toc_init(groups[0], opt['model'])
    logger.info(f"Initial TOC: {len(toc)} entries")

    for i, group in enumerate(groups[1:], 1):
        additional = generate_toc_continue(toc, group, opt['model'])
        logger.info(f"Group {i+1}: +{len(additional)} entries")
        toc.extend(additional)

    convert_physical_index_to_int(toc)
    return toc


def process_toc_no_page_numbers(toc_content: str, page_list: list, start_index: int, opt: dict, logger: SimpleLogger) -> list:
    """Path B: TOC exists but no page numbers — scan pages to find physical indices."""
    logger.info("Path B: TOC without page numbers")

    structure = toc_transformer(toc_content, opt['model'])
    logger.info(f"Transformed TOC: {len(structure)} entries")

    tagged_pages = tag_pages(page_list, start_index)
    token_lengths = [count_tokens(p) for p in tagged_pages]
    groups = page_list_to_group_text(tagged_pages, token_lengths, max_tokens=opt.get('max_token_num_each_node', 6000))

    current_structure = copy.deepcopy(structure)
    for group in groups:
        current_structure = add_page_number_to_toc(group, current_structure, opt['model'])

    convert_physical_index_to_int(current_structure)
    return current_structure


def process_toc_with_page_numbers(toc_content: str, toc_page_list: list, page_list: list, opt: dict, logger: SimpleLogger) -> list:
    """Path A: TOC has page numbers — calculate offset and map to physical indices."""
    logger.info("Path A: TOC with page numbers")

    toc_with_pages = toc_transformer(toc_content, opt['model'])
    logger.info(f"Transformed TOC: {len(toc_with_pages)} entries")

    # Build tagged content from pages right after TOC
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + opt['toc_check_page_num'], len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    # Extract physical indices for a sample of entries
    toc_no_pages = copy.deepcopy(toc_with_pages)
    for item in toc_no_pages:
        item.pop('page', None)

    toc_with_physical = toc_index_extractor(toc_no_pages, main_content, opt['model'])
    convert_physical_index_to_int(toc_with_physical)

    # Calculate page offset
    pairs = extract_matching_page_pairs(toc_with_pages, toc_with_physical, start_page_index)
    offset = calculate_page_offset(pairs)
    logger.info(f"Page offset: {offset}")

    # Apply offset to all entries
    for item in toc_with_pages:
        if item.get('page') is not None and isinstance(item['page'], int):
            item['physical_index'] = item['page'] + offset
            del item['page']

    return toc_with_pages


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — VERIFICATION & SELF-CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

async def check_title_appearance(item: dict, page_list: list, start_index: int, model: str) -> dict:
    """Verify that a section title actually appears on its assigned physical_index page."""
    title = item['title']
    if item.get('physical_index') is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title': title, 'page_number': None}

    page_number = item['physical_index']
    idx = page_number - start_index
    if idx < 0 or idx >= len(page_list):
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title': title, 'page_number': page_number}

    page_text = page_list[idx][0]

    prompt = f"""Your job is to check if the given section title appears or starts in the given page text.
Do fuzzy matching — ignore spacing inconsistencies.

Section title: {title}
Page text: {page_text[:2000]}

Return JSON only:
{{
    "thinking": "<why you think yes or no>",
    "answer": "<yes or no>"
}}
Directly return the JSON. Do not output anything else."""

    response = await llm_acompletion(model, prompt)
    result = extract_json(response)
    answer = result.get('answer', 'no')
    return {'list_index': item.get('list_index'), 'answer': answer, 'title': title, 'page_number': page_number}


async def verify_toc(page_list: list, toc: list, start_index: int, model: str) -> tuple[float, list]:
    """Verify all entries. Returns (accuracy, incorrect_results)."""
    if not toc:
        return 0.0, []

    last_pi = None
    for item in reversed(toc):
        if item.get('physical_index') is not None:
            last_pi = item['physical_index']
            break

    if last_pi is None or last_pi < len(page_list) / 2:
        return 0.0, []

    indexed = []
    for idx, item in enumerate(toc):
        if item.get('physical_index') is not None:
            copy_item = item.copy()
            copy_item['list_index'] = idx
            indexed.append(copy_item)

    tasks = [check_title_appearance(item, page_list, start_index, model) for item in indexed]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r['answer'] == 'yes')
    incorrect = [r for r in results if r['answer'] != 'yes']
    accuracy = correct / len(results) if results else 0.0
    print(f"[PageIndex] Verification accuracy: {accuracy*100:.1f}%")
    return accuracy, incorrect


async def single_toc_item_fixer(title: str, content: str, model: str) -> int | None:
    """Find the correct physical_index for a section by searching tagged pages."""
    prompt = f"""You are given a section title and several tagged pages.
Find the physical index of the page where this section starts.

Pages contain tags like <physical_index_X> to indicate page X.

Section title: {title}
Document pages:
{content}

Return JSON only:
{{
    "thinking": "<which page contains the start of this section>",
    "physical_index": "<physical_index_X>"
}}
Directly return the JSON. Do not output anything else."""

    response = await llm_acompletion(model, prompt)
    result = extract_json(response)
    pi = result.get('physical_index', '')
    import re
    match = re.search(r'physical_index_(\d+)', str(pi))
    return int(match.group(1)) if match else None


async def fix_incorrect_toc(toc: list, page_list: list, incorrect: list, start_index: int, model: str, logger: SimpleLogger) -> tuple[list, list]:
    """Fix incorrect physical indices by narrowing search range."""
    incorrect_indices = {r['list_index'] for r in incorrect}
    end_index = len(page_list) + start_index - 1

    async def fix_one(bad_item):
        li = bad_item['list_index']
        if li < 0 or li >= len(toc):
            return {'list_index': li, 'title': bad_item['title'], 'physical_index': None, 'is_valid': False}

        # Find prev/next correct neighbours
        prev_pi = start_index - 1
        for j in range(li - 1, -1, -1):
            if j not in incorrect_indices:
                pi = toc[j].get('physical_index')
                if pi is not None:
                    prev_pi = pi
                    break

        next_pi = end_index
        for j in range(li + 1, len(toc)):
            if j not in incorrect_indices:
                pi = toc[j].get('physical_index')
                if pi is not None:
                    next_pi = pi
                    break

        # Build tagged content in the search range
        range_content = ""
        for pi in range(prev_pi, next_pi + 1):
            idx = pi - start_index
            if 0 <= idx < len(page_list):
                range_content += f"<physical_index_{pi}>\n{page_list[idx][0]}\n<physical_index_{pi}>\n\n"

        new_pi = await single_toc_item_fixer(bad_item['title'], range_content, model)

        # Verify the fix
        check_item = bad_item.copy()
        check_item['physical_index'] = new_pi
        result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': li,
            'title': bad_item['title'],
            'physical_index': new_pi,
            'is_valid': result['answer'] == 'yes',
        }

    tasks = [fix_one(item) for item in incorrect]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    still_invalid = []
    for result in results:
        if isinstance(result, Exception):
            continue
        if result['is_valid']:
            li = result['list_index']
            if 0 <= li < len(toc):
                toc[li]['physical_index'] = result['physical_index']
        else:
            still_invalid.append(result)

    return toc, still_invalid


async def fix_incorrect_toc_with_retries(toc: list, page_list: list, incorrect: list, start_index: int, model: str, logger: SimpleLogger, max_attempts: int = 3) -> tuple[list, list]:
    current_toc = toc
    current_incorrect = incorrect
    for attempt in range(max_attempts):
        if not current_incorrect:
            break
        logger.info(f"Fix attempt {attempt+1}: {len(current_incorrect)} incorrect entries")
        current_toc, current_incorrect = await fix_incorrect_toc(
            current_toc, page_list, current_incorrect, start_index, model, logger
        )
    return current_toc, current_incorrect


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — LARGE NODE RECURSION
# ══════════════════════════════════════════════════════════════════════════════

async def process_large_node_recursively(node: dict, page_list: list, opt: dict, logger: SimpleLogger):
    """Recursively re-index nodes that are too large."""
    start = node.get('start_index', 1)
    end   = node.get('end_index', len(page_list))
    node_pages = page_list[start - 1:end]
    token_num = sum(p[1] for p in node_pages)

    if (end - start) > opt['max_page_num_each_node'] and token_num >= opt['max_token_num_each_node']:
        logger.info(f"Large node: '{node['title']}' pages {start}-{end} ({token_num} tokens) — recursing")
        sub_toc = process_no_toc(node_pages, start_index=start, opt=opt, logger=logger)

        # Verify and fix sub-toc
        valid_sub = [item for item in sub_toc if item.get('physical_index') is not None]
        accuracy, incorrect = await verify_toc(page_list, valid_sub, start_index=start, model=opt['model'])
        if accuracy < 1.0 and incorrect:
            valid_sub, _ = await fix_incorrect_toc_with_retries(valid_sub, page_list, incorrect, start, opt['model'], logger)

        valid_sub = [item for item in valid_sub if item.get('physical_index') is not None]

        # Avoid duplicating the parent node title
        if valid_sub and valid_sub[0]['title'].strip() == node['title'].strip():
            node['nodes'] = post_processing(valid_sub[1:], end)
        else:
            node['nodes'] = post_processing(valid_sub, end)

    if node.get('nodes'):
        tasks = [process_large_node_recursively(child, page_list, opt, logger) for child in node['nodes']]
        await asyncio.gather(*tasks)

    return node


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def generate_node_summary(node: dict, model: str) -> str:
    """Generate a 2-3 sentence summary for a node."""
    text = node.get('text', '')
    if not text:
        return ''

    prompt = f"""You are given a section of a document. Generate a concise 2-3 sentence summary 
describing the main points covered in this section.

Section text:
{text[:3000]}

Directly return the summary. Do not include any other text."""

    return await llm_acompletion(model, prompt)


async def generate_summaries(structure, model: str):
    """Generate summaries for all nodes in the tree."""
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def meta_processor(page_list: list, mode: str, opt: dict, logger: SimpleLogger,
                          toc_content: str = None, toc_page_list: list = None,
                          start_index: int = 1) -> list:
    """
    Route to the correct processing path and run verification + self-correction.
    Falls back to simpler paths if accuracy is too low (mirrors PageIndex behaviour).
    """
    logger.info(f"meta_processor mode={mode} start_index={start_index}")

    if mode == 'process_toc_with_page_numbers':
        toc = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, opt, logger)
    elif mode == 'process_toc_no_page_numbers':
        toc = process_toc_no_page_numbers(toc_content, page_list, start_index, opt, logger)
    else:
        toc = process_no_toc(page_list, start_index, opt, logger)

    # Filter out entries with no physical_index
    toc = [item for item in toc if item.get('physical_index') is not None]

    # Validate indices don't exceed document length
    max_page = len(page_list) + start_index - 1
    for item in toc:
        if item.get('physical_index', 0) > max_page:
            item['physical_index'] = None
    toc = [item for item in toc if item.get('physical_index') is not None]

    # Verify
    accuracy, incorrect = await verify_toc(page_list, toc, start_index, opt['model'])
    logger.info(f"Verification: {accuracy*100:.1f}% accurate, {len(incorrect)} incorrect")

    if accuracy == 1.0:
        return toc

    if accuracy > 0.6 and incorrect:
        toc, _ = await fix_incorrect_toc_with_retries(toc, page_list, incorrect, start_index, opt['model'], logger)
        return toc

    # Fallback to simpler path
    if mode == 'process_toc_with_page_numbers':
        logger.info("Falling back to: process_toc_no_page_numbers")
        return await meta_processor(page_list, 'process_toc_no_page_numbers', opt, logger,
                                    toc_content=toc_content, toc_page_list=toc_page_list,
                                    start_index=start_index)
    elif mode == 'process_toc_no_page_numbers':
        logger.info("Falling back to: process_no_toc")
        return await meta_processor(page_list, 'process_no_toc', opt, logger, start_index=start_index)
    else:
        logger.info("All paths exhausted — returning best available result")
        return toc


async def tree_parser(page_list: list, opt: dict, logger: SimpleLogger) -> list:
    """Build the full tree: detect TOC path → process → verify → recurse large nodes."""

    # Fast path: small docs skip TOC detection entirely — saves 5-10 LLM calls and runs Path C directly, which is more reliable on short docs anyway
    if len(page_list) <= SMALL_DOC_PAGE_THRESHOLD:
        logger.info(f"Small doc ({len(page_list)} pages) — skipping TOC detection, using Path C directly")
        flat_toc = await meta_processor(page_list, 'process_no_toc', opt, logger)
        flat_toc = add_preface_if_needed(flat_toc)
        valid_toc = [item for item in flat_toc if item.get('physical_index') is not None]
        toc_tree = post_processing(valid_toc, len(page_list))
        tasks = [process_large_node_recursively(node, page_list, opt, logger) for node in toc_tree]
        await asyncio.gather(*tasks)
        return toc_tree

    toc_check = check_toc(page_list, opt, logger)

    if toc_check['toc_content'] and toc_check['page_index_given_in_toc'] == 'yes':
        flat_toc = await meta_processor(
            page_list, 'process_toc_with_page_numbers', opt, logger,
            toc_content=toc_check['toc_content'],
            toc_page_list=toc_check['toc_page_list'],
        )
    elif toc_check['toc_content']:
        flat_toc = await meta_processor(
            page_list, 'process_toc_no_page_numbers', opt, logger,
            toc_content=toc_check['toc_content'],
            toc_page_list=toc_check['toc_page_list'],
        )
    else:
        flat_toc = await meta_processor(
            page_list, 'process_no_toc', opt, logger,
        )

    flat_toc = add_preface_if_needed(flat_toc)

    # Build nested tree
    valid_toc = [item for item in flat_toc if item.get('physical_index') is not None]
    toc_tree = post_processing(valid_toc, len(page_list))

    # Recurse large nodes
    tasks = [process_large_node_recursively(node, page_list, opt, logger) for node in toc_tree]
    await asyncio.gather(*tasks)

    return toc_tree


def build_index(pdf_path: str, model_name: str = "llama3:latest", index_path: str = None,
                add_summaries: bool = False) -> dict:  # summaries off by default for speed
    """
    Main entry point. Build a full PageIndex-style tree from a PDF using Ollama.
    Returns the index dict and optionally saves to index_path.
    """
    logger = SimpleLogger("PageIndex")
    logger.info(f"Starting index build: {pdf_path}")

    opt = {**DEFAULT_CONFIG, 'model': model_name}

    # Extract pages
    page_list = get_page_tokens(pdf_path, model_name)
    logger.info(f"Extracted {len(page_list)} pages, {sum(p[1] for p in page_list)} tokens total")

    # Build tree
    async def run():
        tree = await tree_parser(page_list, opt, logger)

        if opt['if_add_node_id'] == 'yes':
            write_node_id(tree)

        # Always attach raw text so retriever can read sections directly
        add_node_text(tree, page_list)

        if add_summaries:
            await generate_summaries(tree, model_name)

        return tree

    structure = asyncio.run(run())

    index = {
        'doc_name':    get_pdf_name(pdf_path),
        'pdf_path':    pdf_path,
        'filename':    os.path.basename(pdf_path),
        'total_pages': len(page_list),
        'model_used':  model_name,
        'structure':   structure,
    }

    if index_path:
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        logger.info(f"Index saved to {index_path}")

    logger.info("Index build complete!")
    return index

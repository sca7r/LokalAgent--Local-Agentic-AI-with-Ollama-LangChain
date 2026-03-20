"""
agents/planner.py
Planner Agent — breaks a user task into an ordered JSON plan.
Simplified prompt for phi3/smaller models.
"""

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


TOOL_LIST = """web_search     - search the internet
python_repl    - run Python code
read_file      - read a file (input: path)
write_file     - write a file (input: path|||content)
api_call       - HTTP request (input: JSON with url/method)
document_search - search uploaded PDF"""


def run_planner(task: str, model: str, feedback: str = None) -> list:
    """Generate a step-by-step plan for the given task."""
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)

    feedback_str = ""
    if feedback:
        feedback_str = f"\nPrevious attempt failed. Fix this: {feedback}\n"

    prompt = f"""Break this task into steps. Use ONLY these tools:
{TOOL_LIST}

Task: {task}{feedback_str}

Return a JSON array ONLY. Example:
[
  {{"step":1,"description":"search for X","tool":"web_search","input":"X price today"}},
  {{"step":2,"description":"calculate result","tool":"python_repl","input":"result = 100 * 0.1\\nprint(result)"}},
  {{"step":3,"description":"save to file","tool":"write_file","input":"output.txt|||the result is 10"}}
]

Rules:
- Maximum 4 steps
- Every step MUST use a tool from the list above
- Return ONLY the JSON array, no explanation

JSON:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Extract JSON array
    start = content.find('[')
    end   = content.rfind(']')
    if start != -1 and end != -1:
        try:
            plan = json.loads(content[start:end+1])
            if isinstance(plan, list) and len(plan) > 0:
                # Filter out any "none" tool steps
                real_steps = [s for s in plan if s.get('tool','').lower() != 'none']
                if real_steps:
                    return real_steps
        except json.JSONDecodeError:
            pass

    # Fallback: build a simple plan manually based on keywords
    return _fallback_plan(task)


def _fallback_plan(task: str) -> list:
    """Build a basic plan when the LLM fails to produce valid JSON."""
    plan = []
    step = 1
    task_lower = task.lower()

    if any(w in task_lower for w in ['search','find','look up','current','latest','price','news','weather']):
        plan.append({"step": step, "description": "search for information", "tool": "web_search", "input": task})
        step += 1

    if any(w in task_lower for w in ['calculate','compute','math','percent','%','sum','average']):
        # Build a more specific calculation hint
        if '10%' in task or '10 percent' in task_lower:
            calc_hint = "gold_price = REPLACE_WITH_PRICE_FROM_SEARCH\nten_percent = gold_price * 0.10\nprint(f'10% of gold price ${gold_price} = ${ten_percent}')"
        else:
            calc_hint = "# use actual values from search results\nvalue = REPLACE_WITH_VALUE\nresult = value\nprint(result)"
        plan.append({"step": step, "description": "calculate 10% of the price", "tool": "python_repl", "input": calc_hint})
        step += 1

    if any(w in task_lower for w in ['save','write','store','.txt','.py','.json','file']):
        import re
        fname = re.search(r'[\w]+\.\w+', task)
        filename = fname.group(0) if fname else 'output.txt'
        plan.append({"step": step, "description": f"save result to {filename}", "tool": "write_file", "input": f"{filename}|||result from previous steps"})
        step += 1

    if not plan:
        plan.append({"step": 1, "description": task, "tool": "web_search", "input": task})

    return plan

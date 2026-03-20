from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool


def get_code_exec_tool():
    """
    Returns a Python REPL tool that lets the agent write and run Python code.
    Useful for: calculations, data processing, generating files, etc.

    WARNING: This executes real Python code on your machine.
    In production, run inside a sandboxed environment (Docker, etc.)
    """
    repl = PythonREPLTool()

    return Tool(
        name="python_repl",
        func=repl.run,
        description=(
            "Execute Python code. Use this for calculations, data analysis, "
            "generating charts, processing data, or any task that benefits from code. "
            "Input should be valid Python code. "
            "The result of the last expression is returned as output."
        ),
    )
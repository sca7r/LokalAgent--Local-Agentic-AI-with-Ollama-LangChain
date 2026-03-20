from langchain.tools import Tool
import os


def _read_file(path: str) -> str:
    """Read the contents of a file."""
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        return f"Error: File not found at path '{path}'"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content if content else "(empty file)"
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(input_str: str) -> str:
    """
    Write content to a file.
    Input format:  <path>|||<content>
    """
    if "|||" not in input_str:
        return (
            "Error: Input must be in format '<file_path>|||<file_content>'. "
            "Use ||| to separate the path from the content."
        )
    path, content = input_str.split("|||", 1)
    path = path.strip()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file: {e}"


def get_file_tools():
    """Returns file read and write tools."""
    read_tool = Tool(
        name="read_file",
        func=_read_file,
        description=(
            "Read the contents of a file from the local filesystem. "
            "Input: the file path (e.g. './data/report.txt')."
        ),
    )

    write_tool = Tool(
        name="write_file",
        func=_write_file,
        description=(
            "Write content to a file on the local filesystem. "
            "Input format: '<file_path>|||<file_content>' "
            "— use ||| to separate the path from the content. "
            "Example: './output/summary.txt|||Hello, world!'"
        ),
    )

    return [read_tool, write_tool]
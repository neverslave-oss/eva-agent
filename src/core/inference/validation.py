"""Tool-loop validation helpers for the model server.

Extracted from the original model_server.py monolith (refactor). These are the
"did the tool result already answer the query?" heuristics plus their lookup
constants. Stateless — a leaf module, safe to import anywhere without circular
dependency risk.
"""

# Read/query-style informational tools whose successful output can itself be the
# answer to a look-up request. Only these are candidates for validation-stop.
_VALIDATION_INFO_TOOLS = {
    "read_file", "exec_shell", "ls", "glob", "list_dir", "df", "ps",
    "process", "web_search", "list", "grep", "cat", "head", "tail",
}

# Markers that make a query look like a single look-up/read request (vs a
# multi-step task that legitimately needs more tool calls after the first).
_VALIDATION_QUERY_MARKERS = (
    "read ", "ls ", "list ", "show ", "display ", "check ", "how much",
    "how full", "how big", "df ", "ps ", "version", " what", " which",
    " find", " grep", " cat ", " status", " memory", " disk", " file",
    " pid", " process", " available", " installed", " running",
)

# Token patterns that mean the result is an error/banner, never an answer.
_VALIDATION_ERROR_TOKENS = (
    "(error:", "traceback", "permission denied", "command not found",
    "no such file or directory", "failed", "error:", "exception:",
    "unexpected", "illegal instruction", "segmentation",
)


def tool_result_answers_query(query: str, tool_name: str, result: str) -> bool:
    """Lightweight validation: did a successful tool result already answer the
    user's query, so the tool loop can STOP instead of re-calling the same tool?

    This is the "did the result answer the question?" check the loop was
    missing (previously it relied on a blind 3-strike repetition counter). It is
    deliberately conservative and heuristic (no extra LLM round-trip per step):
      - only read/query-style informational tools qualify,
      - result must be substantive and not an error banner,
      - query must look like a single look-up request.
    It returns False on anything ambiguous so legitimate multi-step workflows
    (e.g. read-then-write) are never truncated.
    """
    if tool_name not in _VALIDATION_INFO_TOOLS:
        return False
    if not result or not str(result).strip():
        return False
    rl = str(result).lower()
    if any(t in rl for t in _VALIDATION_ERROR_TOKENS):
        return False
    if len(str(result).strip()) < 4:
        return False
    ql = (query or "").lower()
    return any(m in ql for m in _VALIDATION_QUERY_MARKERS)

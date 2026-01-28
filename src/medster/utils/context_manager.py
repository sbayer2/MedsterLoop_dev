# Context management utilities for preventing token overflow
# Handles truncation and summarization of large tool outputs

import json
from typing import Any, List, Optional

# Approximate tokens per character (conservative estimate for medical text)
CHARS_PER_TOKEN = 3.5

# Maximum tokens for accumulated outputs passed to LLM
MAX_OUTPUT_TOKENS = 50000  # Leave room for system prompt, tools, etc.
MAX_OUTPUT_CHARS = int(MAX_OUTPUT_TOKENS * CHARS_PER_TOKEN)

# Maximum tokens for a single tool output
MAX_SINGLE_OUTPUT_TOKENS = 10000
MAX_SINGLE_OUTPUT_CHARS = int(MAX_SINGLE_OUTPUT_TOKENS * CHARS_PER_TOKEN)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    return int(len(text) / CHARS_PER_TOKEN)


def truncate_output(output: str, max_chars: int = MAX_SINGLE_OUTPUT_CHARS) -> str:
    """
    Truncate a single tool output to prevent token overflow.

    Preserves structure by keeping beginning and end with truncation notice.
    """
    if len(output) <= max_chars:
        return output

    # Keep 40% from start, 40% from end, 20% for truncation notice
    keep_chars = int(max_chars * 0.4)

    start = output[:keep_chars]
    end = output[-keep_chars:]

    truncated_chars = len(output) - (keep_chars * 2)
    truncated_tokens = estimate_tokens(str(truncated_chars))

    return f"{start}\n\n... [TRUNCATED: {truncated_chars} characters (~{truncated_tokens} tokens) removed for context efficiency] ...\n\n{end}"


def summarize_list_result(result: Any, max_items: int = 20) -> Any:
    """
    Summarize list results that contain many items.

    For patient lists, condition lists, etc., keeps first N items
    and adds a summary count.
    """
    if not isinstance(result, dict):
        return result

    summarized = {}

    for key, value in result.items():
        if isinstance(value, list) and len(value) > max_items:
            # Keep first N items and add count
            summarized[key] = value[:max_items]
            summarized[f"{key}_total_count"] = len(value)
            summarized[f"{key}_truncated"] = True
        elif isinstance(value, dict):
            # Recursively summarize nested dicts
            summarized[key] = summarize_list_result(value, max_items)
        else:
            summarized[key] = value

    return summarized


def format_output_for_context(tool_name: str, args: Any, result: Any) -> str:
    """
    Format tool output for context, applying summarization and truncation.
    """
    # First, summarize list results
    if isinstance(result, dict):
        result = summarize_list_result(result)

    # Convert to string
    if isinstance(result, (dict, list)):
        try:
            result_str = json.dumps(result, indent=2, default=str)
        except:
            result_str = str(result)
    else:
        result_str = str(result)

    # Format the output
    output = f"Output of {tool_name} with args {args}: {result_str}"

    # Truncate if too long
    return truncate_output(output)


def manage_context_size(outputs: List[str], max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """
    Manage total context size by truncating older outputs if needed.

    Prioritizes recent outputs over older ones.
    """
    if not outputs:
        return ""

    # Join all outputs
    full_context = "\n".join(outputs)

    if len(full_context) <= max_chars:
        return full_context

    # If over limit, keep most recent outputs and summarize older ones
    recent_outputs = []
    current_size = 0

    # Work backwards (most recent first)
    for output in reversed(outputs):
        output_size = len(output)
        if current_size + output_size <= max_chars * 0.8:  # Keep 80% for recent
            recent_outputs.insert(0, output)
            current_size += output_size
        else:
            break

    # Count truncated outputs
    truncated_count = len(outputs) - len(recent_outputs)

    if truncated_count > 0:
        truncation_notice = f"[CONTEXT MANAGER: {truncated_count} earlier outputs truncated to fit context limit. Keeping {len(recent_outputs)} most recent outputs.]\n\n"
        return truncation_notice + "\n".join(recent_outputs)

    return "\n".join(recent_outputs)


def get_context_stats(outputs: List[str]) -> dict:
    """Get statistics about current context usage."""
    full_context = "\n".join(outputs) if outputs else ""
    char_count = len(full_context)
    token_estimate = estimate_tokens(full_context)

    return {
        "output_count": len(outputs),
        "total_chars": char_count,
        "estimated_tokens": token_estimate,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "utilization_pct": round(token_estimate / MAX_OUTPUT_TOKENS * 100, 1) if MAX_OUTPUT_TOKENS > 0 else 0,
        "at_risk": token_estimate > MAX_OUTPUT_TOKENS * 0.8
    }

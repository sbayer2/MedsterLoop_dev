"""
MedsterLoop - Direct Anthropic SDK Implementation
Replaces LangChain with native Anthropic client for cleaner, faster execution.
"""
import os
import time
import json
from typing import Type, List, Optional, Dict, Any, Union
from pydantic import BaseModel
import anthropic

from medster.prompts import DEFAULT_SYSTEM_PROMPT


# Initialize Anthropic client
_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """Get or create Anthropic client singleton."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


# Model name mapping
MODEL_MAPPING = {
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-haiku-4": "claude-haiku-4-20250107",
}


def convert_tools_to_anthropic_format(tools: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert tool objects to Anthropic's tool format.

    Supports both:
    - LangChain-style tools (for backward compatibility)
    - Dict-based tool definitions (new style)
    """
    anthropic_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            # Already in dict format
            anthropic_tools.append(tool)
        elif hasattr(tool, 'name') and hasattr(tool, 'description'):
            # LangChain-style tool object
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            # Extract schema from args_schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
                tool_def["input_schema"]["properties"] = schema.get("properties", {})
                tool_def["input_schema"]["required"] = schema.get("required", [])
            anthropic_tools.append(tool_def)
    return anthropic_tools


def call_llm(
    prompt: str,
    model: str = "claude-sonnet-4.5",
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[Any]] = None,
    images: Optional[List[str]] = None,
    max_tokens: int = 4096,
) -> Union[Dict[str, Any], Any]:
    """
    Call Claude LLM with the given prompt and configuration.

    Returns a dict with:
    - content: str (text response)
    - tool_calls: List[Dict] (if tools were called)
    - stop_reason: str
    - usage: Dict (token counts)

    Or if output_schema is provided, returns the parsed Pydantic object.
    """
    client = get_client()
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    anthropic_model = MODEL_MAPPING.get(model, "claude-sonnet-4-5-20250929")

    # Build message content
    if images:
        # Multimodal message with images
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img_base64 in images:
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            })
        messages = [{"role": "user", "content": content_parts}]
    else:
        messages = [{"role": "user", "content": prompt}]

    # Build API call kwargs
    api_kwargs = {
        "model": anthropic_model,
        "max_tokens": max_tokens,
        "system": final_system_prompt,
        "messages": messages,
    }

    # Add tools if provided
    if tools:
        api_kwargs["tools"] = convert_tools_to_anthropic_format(tools)

    # Retry logic for transient errors and rate limits
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**api_kwargs)
            break
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = 10 * (2 ** attempt)
            print(f"Rate limit hit. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
        except anthropic.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(1 * (2 ** attempt))

    # Parse response
    result = {
        "content": "",
        "tool_calls": [],
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    }

    for block in response.content:
        if block.type == "text":
            result["content"] += block.text
        elif block.type == "tool_use":
            result["tool_calls"].append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })

    # If output_schema is provided, parse the response
    if output_schema:
        try:
            # Try to parse JSON from the response
            text = result["content"]
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            return output_schema(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Return None if parsing fails - let caller handle it
            print(f"Warning: Failed to parse structured output: {e}")
            return None

    return result


def call_llm_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Any],
    model: str = "claude-sonnet-4.5",
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Call Claude with a conversation history and tools.
    This is the primary function for the event loop pattern.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: List of tool definitions
        model: Model name
        system_prompt: System prompt
        max_tokens: Max tokens for response

    Returns:
        Dict with content, tool_calls, stop_reason, usage
    """
    client = get_client()
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    anthropic_model = MODEL_MAPPING.get(model, "claude-sonnet-4-5-20250929")

    api_kwargs = {
        "model": anthropic_model,
        "max_tokens": max_tokens,
        "system": final_system_prompt,
        "messages": messages,
        "tools": convert_tools_to_anthropic_format(tools),
    }

    # Retry logic
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**api_kwargs)
            break
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = 10 * (2 ** attempt)
            print(f"Rate limit hit. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        except anthropic.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(1 * (2 ** attempt))

    # Parse response
    result = {
        "content": "",
        "tool_calls": [],
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    }

    for block in response.content:
        if block.type == "text":
            result["content"] += block.text
        elif block.type == "tool_use":
            result["tool_calls"].append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })

    return result

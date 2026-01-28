"""
MedsterLoop Agent - Single-Turn Tool Call Architecture

This replaces the multi-step LangGraph-style state machine with a simpler
single-turn tool call loop. The model decides what to do, we execute tools,
and loop until the model produces a final response without tool calls.

Key differences from Medster_dev:
- No separate task planning step (model decides flow)
- No separate validation/meta-validation LLM calls
- No optimize_tool_args step (model provides correct args directly)
- Single unified system prompt
- Model naturally decides when analysis is complete

This reduces 6 LLM calls per iteration to 1.
"""
from typing import List, Optional, Dict, Any
import re

from medster.model import call_llm_with_tools, call_llm
from medster.prompts import UNIFIED_SYSTEM_PROMPT, get_answer_system_prompt
from medster.schemas import Answer
from medster.tools import TOOLS, get_tool_by_name, execute_tool
from medster.utils.logger import Logger
from medster.utils.ui import show_progress
from medster.utils.context_manager import format_output_for_context, manage_context_size


class Agent:
    """
    MedsterLoop Agent using single-turn tool call pattern.

    The agent loop is simple:
    1. Send query + conversation history to model
    2. If model returns tool calls, execute them and add results to history
    3. Loop until model returns a response without tool calls
    4. Return the final response
    """

    def __init__(self, max_iterations: int = 20):
        self.logger = Logger()
        self.max_iterations = max_iterations
        self.uploaded_content: Optional[str] = None
        self.uploaded_filename: Optional[str] = None

    def _extract_uploaded_content(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """Extract uploaded file content from query if present."""
        pattern = r'---\s*(?:Attached\s+)?File:\s*([^-\n]+)\s*---\s*([\s\S]+?)(?:\[\.\.\.\s*FILE TRUNCATED|$)'
        match = re.search(pattern, query)

        if match:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            self.logger._log(f"Extracted uploaded content: {filename} ({len(content)} chars)")
            return content, filename

        return None, None

    def _build_tool_result_content(self, tool_id: str, tool_name: str, result: Any, is_error: bool = False) -> Dict[str, Any]:
        """Build Anthropic-format tool result content block."""
        if is_error:
            content = f"Error: {result}"
        else:
            content = format_output_for_context(tool_name, {}, result)

        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content,
        }

    @show_progress("Analyzing clinical data...", "")
    def _agent_turn(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single agent turn - call model and return response."""
        return call_llm_with_tools(
            messages=messages,
            tools=TOOLS,
            system_prompt=UNIFIED_SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """
        Execute the agent loop to process a clinical query.

        Args:
            query: The user's clinical analysis query

        Returns:
            str: A comprehensive clinical analysis response
        """
        self.logger.log_user_query(query)

        # Extract uploaded file content if present
        self.uploaded_content, self.uploaded_filename = self._extract_uploaded_content(query)
        if self.uploaded_content:
            self.logger._log(f"📎 Detected uploaded file: {self.uploaded_filename}")

        # Initialize conversation with user query
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": query}
        ]

        # Track tool outputs for final summary
        tool_outputs: List[str] = []
        iteration = 0
        total_tokens = {"input": 0, "output": 0}

        # Main agent loop
        while iteration < self.max_iterations:
            iteration += 1
            self.logger._log(f"🔄 Iteration {iteration}/{self.max_iterations}")

            # Get model response
            response = self._agent_turn(messages)

            # Track token usage
            total_tokens["input"] += response["usage"]["input_tokens"]
            total_tokens["output"] += response["usage"]["output_tokens"]

            # Check if model is done (no tool calls)
            if not response["tool_calls"]:
                self.logger._log(f"✅ Model completed analysis after {iteration} iterations")
                self.logger._log(f"📊 Total tokens: {total_tokens['input']} in / {total_tokens['output']} out")

                # Return model's final response
                final_response = response["content"]
                if final_response:
                    self.logger.log_summary(final_response)
                    return final_response
                else:
                    # If model returns empty response, generate summary from tool outputs
                    return self._generate_summary(query, tool_outputs)

            # Build assistant message with tool calls
            assistant_content = []
            if response["content"]:
                assistant_content.append({"type": "text", "text": response["content"]})

            for tool_call in response["tool_calls"]:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call["args"],
                })

            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tool calls and collect results
            tool_results = []
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                self.logger._log(f"🔧 Executing tool: {tool_name}")

                # Inject uploaded content for code generation tool
                if tool_name == "generate_and_run_analysis" and self.uploaded_content:
                    tool_args["uploaded_content"] = self.uploaded_content
                    self.logger._log(f"📎 Injecting uploaded content into {tool_name}")

                try:
                    result = execute_tool(tool_name, tool_args)
                    self.logger.log_tool_run(tool_args, result)

                    # Format and store output
                    output = format_output_for_context(tool_name, tool_args, result)
                    tool_outputs.append(output)

                    tool_results.append(
                        self._build_tool_result_content(tool_id, tool_name, result)
                    )
                except Exception as e:
                    self.logger._log(f"❌ Tool execution failed: {e}")
                    error_output = f"Error from {tool_name}: {e}"
                    tool_outputs.append(error_output)

                    tool_results.append(
                        self._build_tool_result_content(tool_id, tool_name, str(e), is_error=True)
                    )

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})

            # Context management - truncate old tool results if context is getting large
            if len(tool_outputs) > 10:
                tool_outputs = tool_outputs[-10:]  # Keep last 10 outputs

        # Max iterations reached - generate summary from what we have
        self.logger._log(f"⚠️ Max iterations ({self.max_iterations}) reached")
        return self._generate_summary(query, tool_outputs)

    @show_progress("Generating clinical summary...", "Analysis complete")
    def _generate_summary(self, query: str, tool_outputs: List[str]) -> str:
        """Generate final clinical summary from collected tool outputs."""
        all_results = manage_context_size(tool_outputs) if tool_outputs else "No clinical data was collected."

        summary_prompt = f"""
Original clinical query: "{query}"

Clinical data and results collected:
{all_results}

Based on the data above, provide a comprehensive clinical analysis.
Include specific values, reference ranges, trends, and clinical implications.
Flag any critical findings that require immediate attention.
"""

        try:
            answer_obj = call_llm(
                summary_prompt,
                system_prompt=get_answer_system_prompt(),
                output_schema=Answer
            )

            if answer_obj and hasattr(answer_obj, 'answer'):
                return answer_obj.answer
            elif answer_obj:
                return str(answer_obj)
            else:
                # Fallback: return raw tool outputs
                return f"Clinical analysis completed.\n\nCollected data:\n{all_results[:2000]}..."

        except Exception as e:
            self.logger._log(f"❌ Summary generation failed: {e}")
            return f"Clinical analysis completed with errors.\n\nError: {str(e)}\n\nCollected data:\n{all_results[:1000]}..."

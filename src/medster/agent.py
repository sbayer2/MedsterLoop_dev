from typing import List, Optional
import re

from langchain_core.messages import AIMessage

from medster.model import call_llm
from medster.prompts import (
    ACTION_SYSTEM_PROMPT,
    get_answer_system_prompt,
    PLANNING_SYSTEM_PROMPT,
    get_tool_args_system_prompt,
    VALIDATION_SYSTEM_PROMPT,
    META_VALIDATION_SYSTEM_PROMPT,
)
from medster.schemas import Answer, IsDone, OptimizedToolArgs, Task, TaskList
from medster.tools import TOOLS
from medster.utils.logger import Logger
from medster.utils.ui import show_progress
from medster.utils.context_manager import (
    format_output_for_context,
    manage_context_size,
    get_context_stats
)


class Agent:
    def __init__(self, max_steps: int = 20, max_steps_per_task: int = 5):
        self.logger = Logger()
        self.max_steps = max_steps            # global safety cap
        self.max_steps_per_task = max_steps_per_task
        self.uploaded_content: Optional[str] = None  # Stores uploaded file content for code generation
        self.uploaded_filename: Optional[str] = None  # Stores uploaded filename

    def _extract_uploaded_content(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """Extract uploaded file content from query if present.

        Returns:
            Tuple of (content, filename) or (None, None) if no uploaded content
        """
        # Pattern: --- Attached File: filename --- or --- File: filename ---
        # Content follows until [... FILE TRUNCATED or end of string
        pattern = r'---\s*(?:Attached\s+)?File:\s*([^-\n]+)\s*---\s*([\s\S]+?)(?:\[\.\.\.\s*FILE TRUNCATED|$)'
        match = re.search(pattern, query)

        if match:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            self.logger._log(f"Extracted uploaded content: {filename} ({len(content)} chars)")
            return content, filename

        return None, None

    # ---------- task planning ----------
    @show_progress("Planning clinical analysis...", "Tasks planned")
    def plan_tasks(self, query: str) -> List[Task]:
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in TOOLS])
        prompt = f"""
        Given the clinical query: "{query}",
        Create a list of tasks to be completed.
        Example: {{"tasks": [{{"id": 1, "description": "some task", "done": false}}]}}
        """
        system_prompt = PLANNING_SYSTEM_PROMPT.format(tools=tool_descriptions)
        try:
            response = call_llm(prompt, system_prompt=system_prompt, output_schema=TaskList)
            tasks = response.tasks
        except Exception as e:
            self.logger._log(f"Planning failed: {e}")
            tasks = [Task(id=1, description=query, done=False)]

        task_dicts = [task.dict() for task in tasks]
        self.logger.log_task_list(task_dicts)
        return tasks

    # ---------- ask LLM what to do ----------
    @show_progress("Analyzing...", "")
    def ask_for_actions(self, task_desc: str, last_outputs: str = "") -> AIMessage:
        # Check if this is an MCP task that requires mandatory tool call
        is_mcp_task = any(keyword in task_desc.lower() for keyword in ["mcp server", "mcp", "analyze_medical_document", "submit to mcp", "send to mcp"])

        prompt = f"""
        We are working on: "{task_desc}".
        Here is a history of tool outputs from the session so far: {last_outputs}

        Based on the task and the outputs, what should be the next step?
        """

        # Add explicit MCP reminder if this is an MCP task
        if is_mcp_task and "analyze_medical_document" not in last_outputs:
            prompt += """

        **CRITICAL REMINDER**: This task REQUIRES calling the analyze_medical_document tool to send data to the MCP server.
        Simply having the data in previous outputs is NOT sufficient - you MUST call analyze_medical_document with that data.
        Extract the clinical note/document text from the previous outputs and pass it to analyze_medical_document.
        """

        try:
            ai_message = call_llm(prompt, system_prompt=ACTION_SYSTEM_PROMPT, tools=TOOLS)
            return ai_message
        except Exception as e:
            self.logger._log(f"ask_for_actions failed: {e}")
            # Return special marker to indicate failure (not completion)
            return AIMessage(content="AGENT_ERROR: " + str(e))

    # ---------- ask LLM if task is done ----------
    @show_progress("Checking if task is complete...", "")
    def ask_if_done(self, task_desc: str, recent_results: str) -> bool:
        prompt = f"""
        We were trying to complete the task: "{task_desc}".
        Here is a history of tool outputs from the session so far: {recent_results}

        Is the task done?
        """
        try:
            resp = call_llm(prompt, system_prompt=VALIDATION_SYSTEM_PROMPT, output_schema=IsDone)
            return resp.done
        except:
            return False

    # ---------- ask LLM if main goal is achieved ----------
    @show_progress("Checking if analysis is complete...", "")
    def is_goal_achieved(self, query: str, task_outputs: list, tasks: list = None) -> bool:
        """Check if the overall goal is achieved based on all session outputs and task completion."""
        all_results = "\n\n".join(task_outputs)

        # Format task plan for meta-validator
        task_plan = ""
        if tasks:
            task_list = []
            for i, task in enumerate(tasks, 1):
                status = "✓ COMPLETED" if task.done else "✗ NOT COMPLETED"
                task_list.append(f"{i}. {status}: {task.description}")
            task_plan = f"""
Task Plan:
{chr(10).join(task_list)}
"""

        prompt = f"""
        Original clinical query: "{query}"
{task_plan}
        Data and results collected from tools so far:
        {all_results}

        Based on the task plan and data above, is the original clinical query sufficiently answered?
        """
        try:
            resp = call_llm(prompt, system_prompt=META_VALIDATION_SYSTEM_PROMPT, output_schema=IsDone)
            return resp.done
        except Exception as e:
            self.logger._log(f"Meta-validation failed: {e}")
            return False

    # ---------- optimize tool arguments ----------
    @show_progress("Optimizing data request...", "")
    def optimize_tool_args(self, tool_name: str, initial_args: dict, task_desc: str) -> dict:
        """Optimize tool arguments based on task requirements."""
        tool = next((t for t in TOOLS if t.name == tool_name), None)
        if not tool:
            return initial_args

        tool_description = tool.description
        tool_schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}

        prompt = f"""
        Task: "{task_desc}"
        Tool: {tool_name}
        Tool Description: {tool_description}
        Tool Parameters: {tool_schema}
        Initial Arguments: {initial_args}

        Review the task and optimize the arguments to ensure all relevant parameters are used correctly.
        Pay special attention to filtering parameters that would help narrow down results to match the task.
        """
        try:
            response = call_llm(prompt, model="claude-sonnet-4.5", system_prompt=get_tool_args_system_prompt(), output_schema=OptimizedToolArgs)
            if isinstance(response, dict):
                return response if response else initial_args
            return response.arguments
        except Exception as e:
            self.logger._log(f"Argument optimization failed: {e}, using original args")
            return initial_args

    # ---------- tool execution ----------
    def _execute_tool(self, tool, tool_name: str, inp_args):
        """Execute a tool with progress indication."""
        @show_progress(f"Fetching {tool_name}...", "")
        def run_tool():
            return tool.run(inp_args)
        return run_tool()

    # ---------- confirm action ----------
    def confirm_action(self, tool: str, input_str: str) -> bool:
        # In production, could add safety checks for sensitive operations
        return True

    # ---------- main loop ----------
    def run(self, query: str):
        """
        Executes the main agent loop to process a clinical query.

        Args:
            query (str): The user's clinical analysis query.

        Returns:
            str: A comprehensive clinical analysis response.
        """
        self.logger.log_user_query(query)

        # Extract uploaded file content if present in query
        self.uploaded_content, self.uploaded_filename = self._extract_uploaded_content(query)
        if self.uploaded_content:
            self.logger._log(f"📎 Detected uploaded file: {self.uploaded_filename}")

        step_count = 0
        last_actions = []
        task_outputs = []

        # 1. Decompose the clinical query into tasks
        tasks = self.plan_tasks(query)

        if not tasks:
            answer = self._generate_answer(query, task_outputs)
            self.logger.log_summary(answer)
            return answer

        # 2. Loop through tasks until complete or max steps reached
        while any(not t.done for t in tasks):
            if step_count >= self.max_steps:
                self.logger._log("Global max steps reached - stopping to prevent runaway loop.")
                break

            task = next(t for t in tasks if not t.done)
            self.logger.log_task_start(task.description)

            per_task_steps = 0
            task_step_outputs = []

            while per_task_steps < self.max_steps_per_task:
                if step_count >= self.max_steps:
                    self.logger._log("Global max steps reached - stopping.")
                    return

                # Pass outputs with context management to prevent token overflow
                # Uses truncation and prioritizes recent outputs
                all_session_outputs = manage_context_size(task_outputs + task_step_outputs)

                # Log context stats periodically for debugging
                stats = get_context_stats(task_outputs + task_step_outputs)
                if stats["at_risk"]:
                    self.logger._log(f"Context warning: {stats['estimated_tokens']}/{stats['max_tokens']} tokens ({stats['utilization_pct']}%)")

                ai_message = self.ask_for_actions(task.description, last_outputs=all_session_outputs)

                # Check for agent error (e.g., token overflow)
                if hasattr(ai_message, 'content') and isinstance(ai_message.content, str) and ai_message.content.startswith("AGENT_ERROR:"):
                    self.logger._log(f"Task failed due to agent error - NOT marking as complete")
                    # Don't mark as done - break to try next task or finish
                    break

                if not ai_message.tool_calls:
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

                for tool_call in ai_message.tool_calls:
                    if step_count >= self.max_steps:
                        break

                    tool_name = tool_call["name"]
                    initial_args = tool_call["args"]

                    # Validate that we have non-empty arguments
                    if not initial_args or initial_args == {}:
                        self.logger._log(f"⚠️  Skipping {tool_name} - LLM returned empty arguments. This often happens with complex tools like generate_and_run_analysis.")
                        self.logger._log(f"💡 Suggestion: Try a simpler query or use basic tools instead of code generation.")
                        error_output = f"Tool {tool_name} was called with empty arguments - skipped. The model may be struggling with this complex tool."
                        task_outputs.append(error_output)
                        task_step_outputs.append(error_output)
                        continue

                    optimized_args = self.optimize_tool_args(tool_name, initial_args, task.description)

                    # Inject uploaded_content for generate_and_run_analysis if file was uploaded
                    if tool_name == "generate_and_run_analysis" and self.uploaded_content:
                        optimized_args["uploaded_content"] = self.uploaded_content
                        self.logger._log(f"📎 Injecting uploaded content ({len(self.uploaded_content)} chars) into {tool_name}")

                    action_sig = f"{tool_name}:{optimized_args}"

                    # Loop detection - abort if same action repeated 3+ times consecutively
                    last_actions.append(action_sig)
                    if len(last_actions) > 4:
                        last_actions = last_actions[-4:]
                    if len(set(last_actions)) == 1 and len(last_actions) >= 3:
                        self.logger._log("Detected repeating action (3+ identical calls) - aborting to avoid loop.")
                        error_msg = f"Analysis stopped: The system detected a repeating pattern and aborted to prevent an infinite loop. The task '{task.description}' may need to be reformulated or broken down differently."
                        task_outputs.append(error_msg)
                        # Generate answer with what we have so far
                        answer = self._generate_answer(query, task_outputs)
                        self.logger.log_summary(answer)
                        return answer

                    tool_to_run = next((t for t in TOOLS if t.name == tool_name), None)
                    if tool_to_run and self.confirm_action(tool_name, str(optimized_args)):
                        try:
                            result = self._execute_tool(tool_to_run, tool_name, optimized_args)
                            self.logger.log_tool_run(optimized_args, result)
                            # Use context manager to format and truncate large outputs
                            output = format_output_for_context(tool_name, optimized_args, result)
                            task_outputs.append(output)
                            task_step_outputs.append(output)
                        except Exception as e:
                            self.logger._log(f"Tool execution failed: {e}")
                            error_output = f"Error from {tool_name} with args {optimized_args}: {e}"
                            task_outputs.append(error_output)
                            task_step_outputs.append(error_output)
                    else:
                        self.logger._log(f"Invalid tool: {tool_name}")

                    step_count += 1
                    per_task_steps += 1

                # Check if MCP task actually called the required tool
                is_mcp_task = any(kw in task.description.lower() for kw in ["mcp", "analyze_medical_document"])
                mcp_tool_called = any("analyze_medical_document" in output for output in task_step_outputs)

                if is_mcp_task and not mcp_tool_called:
                    self.logger._log(f"MCP task did not call analyze_medical_document - NOT marking as complete")
                    # Don't mark as done - the tool wasn't called
                    break

                # Check if comorbidity/comprehensive task used shallow batch analysis
                is_comorbidity_task = any(kw in task.description.lower() for kw in [
                    "comorbid", "comorbidity", "comprehensive", "review each", "analyze each",
                    "medication", "labs", "vitals", "treatment"
                ])
                used_batch_conditions = any("analyze_batch_conditions" in output for output in task_step_outputs)
                used_code_generation = any("generate_and_run_analysis" in output for output in task_step_outputs)

                if is_comorbidity_task and used_batch_conditions and not used_code_generation:
                    self.logger._log(f"Comorbidity task used analyze_batch_conditions (shallow) - needs generate_and_run_analysis for full patient records")
                    # Don't mark as done - need to use code generation instead
                    break

                if self.ask_if_done(task.description, "\n".join(task_step_outputs)):
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

            if task.done and self.is_goal_achieved(query, task_outputs, tasks):
                self.logger._log("Clinical analysis complete. Generating summary.")
                break

        answer = self._generate_answer(query, task_outputs)
        self.logger.log_summary(answer)
        return answer

    # ---------- answer generation ----------
    @show_progress("Generating clinical summary...", "Analysis complete")
    def _generate_answer(self, query: str, task_outputs: list) -> str:
        """Generate the final clinical analysis based on collected data."""
        # Apply context management to prevent token overflow in final summary
        all_results = manage_context_size(task_outputs) if task_outputs else "No clinical data was collected."
        answer_prompt = f"""
        Original clinical query: "{query}"

        Clinical data and results collected:
        {all_results}

        Based on the data above, provide a comprehensive clinical analysis.
        Include specific values, reference ranges, trends, and clinical implications.
        Flag any critical findings that require immediate attention.
        """
        try:
            answer_obj = call_llm(answer_prompt, system_prompt=get_answer_system_prompt(), output_schema=Answer)
            
            # Check if we got a valid response
            if answer_obj is None:
                self.logger._log("⚠️  Answer generation returned None - falling back to direct text response")
                # Fallback: try without structured output
                fallback_response = call_llm(answer_prompt, system_prompt=get_answer_system_prompt())
                if fallback_response and hasattr(fallback_response, 'content'):
                    return fallback_response.content
                else:
                    return f"Analysis completed but summary generation failed. Raw data: {all_results[:500]}..."
            
            # Extract answer from structured response
            if hasattr(answer_obj, 'answer'):
                return answer_obj.answer
            else:
                self.logger._log(f"⚠️  Unexpected answer object type: {type(answer_obj)}")
                return str(answer_obj) if answer_obj else "Analysis completed but summary generation failed."
                
        except Exception as e:
            self.logger._log(f"❌ Answer generation failed: {e}")
            # Return a basic summary of the collected data
            return f"Clinical analysis completed with errors during summary generation.\n\nError: {str(e)}\n\nCollected data summary:\n{all_results[:1000]}..."


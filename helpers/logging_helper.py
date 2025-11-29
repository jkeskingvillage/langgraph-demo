import os
import weave
from typing import Dict, Any, Optional
from langsmith import traceable
from langsmith.client import Client

# Optional: Set LangSmith environment variables
if os.getenv("LANGCHAIN_TRACING_V2", None) is None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

if os.getenv("LANGCHAIN_PROJECT", None) is None:
    os.environ["LANGCHAIN_PROJECT"] = "ai-agent-demo"

# Initialize Weave
weave.init(project_name=os.getenv("LANGCHAIN_PROJECT"))

# Initialize LangSmith client
client = Client()


class TraceLogger:
    def __init__(self):
        self.traces = []

    @weave.op()
    @traceable(name="agent_execution")
    def log_agent_execution(
        self, input_text: str, output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log the complete agent execution including reasoning, tool calls, and outputs"""
        trace_data = {
            "input": input_text,
            "output": output.get("output", ""),
            "intermediate_steps": output.get("intermediate_steps", []),
            "tool_calls": self._extract_tool_calls(output),
            "reasoning": self._extract_reasoning(output),
            "ollama_usage": self._extract_ollama_usage(output),
        }

        return trace_data

    def _extract_tool_calls(self, output: Dict[str, Any]) -> list:
        """Extract tool calls from intermediate steps"""
        tool_calls = []
        for step in output.get("intermediate_steps", []):
            if len(step) >= 2:
                action = step[0]
                tool_calls.append(
                    {
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                        "observation": step[1],
                    }
                )
        return tool_calls

    def _extract_reasoning(self, output: Dict[str, Any]) -> list:
        """Extract reasoning from intermediate steps"""
        reasoning = []
        for step in output.get("intermediate_steps", []):
            if len(step) >= 2:
                action = step[0]
                reasoning.append(
                    {
                        "thought": getattr(action, "log", ""),
                        "action": action.tool,
                        "input": action.tool_input,
                    }
                )
        return reasoning

    def _extract_ollama_usage(self, output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract Ollama token usage if available"""
        try:
            # Check if we have a final output that contains usage information
            final_output = output.get("output", "")

            # Look for usage information in intermediate steps
            intermediate_steps = output.get("intermediate_steps", [])

            # Check if any step contains usage information
            for step in intermediate_steps:
                if len(step) >= 2:
                    # The second element of each step is typically the observation
                    # which may contain token usage information
                    observation = step[1]
                    if (
                        isinstance(observation, dict)
                        and "prompt_eval_count" in observation
                    ):
                        return {
                            "prompt_tokens": observation.get("prompt_eval_count", 0),
                            "completion_tokens": observation.get("eval_count", 0),
                            "total_tokens": observation.get("prompt_eval_count", 0)
                            + observation.get("eval_count", 0),
                        }
                    elif (
                        isinstance(observation, str)
                        and "prompt_eval_count" in observation
                    ):
                        # If it's a string, try to parse it
                        import re

                        prompt_match = re.search(
                            r"prompt_eval_count:\s*(\d+)", observation
                        )
                        eval_match = re.search(r"eval_count:\s*(\d+)", observation)
                        if prompt_match or eval_match:
                            return {
                                "prompt_tokens": (
                                    int(prompt_match.group(1)) if prompt_match else 0
                                ),
                                "completion_tokens": (
                                    int(eval_match.group(1)) if eval_match else 0
                                ),
                                "total_tokens": (
                                    int(prompt_match.group(1)) if prompt_match else 0
                                )
                                + (int(eval_match.group(1)) if eval_match else 0),
                            }

            # Check if the final output contains usage information
            if isinstance(final_output, dict):
                if "prompt_eval_count" in final_output:
                    return {
                        "prompt_tokens": final_output.get("prompt_eval_count", 0),
                        "completion_tokens": final_output.get("eval_count", 0),
                        "total_tokens": final_output.get("prompt_eval_count", 0)
                        + final_output.get("eval_count", 0),
                    }

            # If we can't find specific usage data, return None
            return None

        except Exception:
            # If there's any error in parsing, return None
            return None


# Global logger instance
trace_logger = TraceLogger()

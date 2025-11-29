import weave
from langchain_core.tools import Tool
import re

from generators.math_problem_generator import safe_eval


def filter_math_symbols(text):
    # Remove all non-math symbols and numbers
    pattern = r"(?:[0-9]+(?:\.[0-9]+)?|[+\-*/()])"
    filtered = re.findall(pattern, text)
    return "".join(filtered)


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="A tool for calculating mathematical expressions",
            func=self._run,
        )

    @weave.op()
    def _run(self, expression: str) -> str:
        """Execute mathematical expression"""
        try:
            expression = filter_math_symbols(expression[:])
            # Remove any non-mathematical characters except numbers, operators, and parentheses
            result = safe_eval(expression)

            return f"= {result}\n"
        except Exception as e:
            return f"Error calculating: {str(e)}"

    async def _arun(self, expression: str):
        raise NotImplementedError("Async not implemented")

"""Safe calculator tool implementation using AST parsing."""

import ast
import math
import operator
import re
from app.tools.base_tool import BaseTool

# Safe math functions mapping
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
}

# Safe operator mapping
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_eval(node) -> float | int:
    """Recursively evaluate an AST expression node safely."""
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.Num):  # Python < 3.8 fallback
        return node.n
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "math":
                func_name = node.func.attr
        
        if func_name in _FUNCTIONS and len(node.args) == 1:
            arg_val = safe_eval(node.args[0])
            return _FUNCTIONS[func_name](arg_val)
        raise ValueError(f"Unsupported function: {func_name}")
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Tool for safely executing basic mathematical calculations and expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Execute arithmetic expressions and mathematical calculations."

    def can_handle(self, query: str) -> bool:
        # Intent router does the selection, but stub can_handle returns True for arithmetic queries
        return bool(re.match(r'^[\d\s+\-*/()^.%]+$', query.strip()))

    def available(self) -> bool:
        return True

    def execute(self, query: str) -> dict:
        expr = query.strip().rstrip("?").strip()
        
        # Clean query by removing common prefix keywords
        for prefix in ["calculate", "what is", "compute", "value of"]:
            if expr.lower().startswith(prefix):
                expr = expr[len(prefix):].strip()
                
        # Resolve percent expressions: e.g. "25% of 400" -> "(25/100)*400"
        expr = re.sub(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', r'(\1/100)*\2', expr)

        try:
            tree = ast.parse(expr, mode="eval")
            result = safe_eval(tree)
            return {
                "success": True,
                "tool": self.name,
                "data": result,
                "content": str(result),
                "sources": [],
                "metadata": {"expr": expr}
            }
        except Exception as exc:
            return {
                "success": False,
                "tool": self.name,
                "error": f"Failed to evaluate expression: {exc}",
                "content": "",
                "sources": [],
                "metadata": {}
            }

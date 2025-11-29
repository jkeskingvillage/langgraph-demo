import hashlib
import random
import string
from typing import List, Tuple
import ast
import operator

import weave


def generate_two_integers() -> Tuple[int, int]:
    """
    Randomly generate two integers less than 1000.

    Returns:
        tuple: A tuple containing two integers (int1, int2) where both are < 1000
    """
    int1 = random.randint(0, 999)
    int2 = random.randint(0, 999)
    return (int1, int2)


def generate_operator():
    """
    Randomly returns one of the four arithmetic operators: "+", "-", "*", or "/".

    Returns:
        str: A randomly selected arithmetic operator
    """
    operators = ["+", "-", "*", "/"]
    return random.choice(operators)


def safe_eval(expression):
    """
    Safely evaluate a mathematical expression string.
    """
    # Define allowed operations
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        return eval_node(tree.body)
    except Exception as e:
        return f"Error: {e}"


def get_problem() -> Tuple[str, str]:
    operator_dict = {"+": "plus", "-": "minus", "*": "multiply by", "/": "divided by"}
    num_0, num_1 = generate_two_integers()
    op = generate_operator()
    problem_str = f"what is {num_0} {operator_dict[op]} {num_1}?"
    problem_expression = f"{num_0} {op} {num_1}"
    answer = safe_eval(problem_expression)

    return problem_str, str(answer)


def generate_random_string(length: int = 32) -> str:
    """
    Generate a random string of specified length.

    Args:
        length (int): Length of the random string to generate (default: 32)

    Returns:
        str: Random string containing letters and digits
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def get_random_string_hash_simple(length: int = 32) -> str:
    """
    Simple version that generates a random string and returns its hash.

    Args:
        length (int): Length of the random string to generate (default: 32)

    Returns:
        str: SHA-256 hash of the random string
    """
    random_string = generate_random_string(length)
    hash_object = hashlib.sha256(random_string.encode())
    return hash_object.hexdigest()


def get_problems(num_problems: int) -> Tuple[List[str], List[str]]:
    problems = []
    answers = []
    for _ in range(num_problems):
        problem_str, answer = get_problem()
        problems.append(problem_str)
        answers.append(answer)

    examples = [
        {"id": str(idx), "sentence": content[0], "target": content[1]}
        for idx, content in enumerate(zip(problems, answers))
    ]
    dataset = weave.Dataset(name=get_random_string_hash_simple(), rows=examples)
    weave.publish(dataset)

    return problems, answers

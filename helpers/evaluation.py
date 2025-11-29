import weave
from typing import Dict, Any, List


@weave.op()
def calculate_accuracy_evaluation(response: str, expected: str) -> Dict[str, Any]:
    """Custom evaluation for calculation accuracy"""
    try:
        # Extract numbers from response and expected
        response_numbers = [
            float(x)
            for x in response.split()
            if x.replace(".", "").isdigit()
            or (x.startswith("-") and x[1:].replace(".", "").isdigit())
        ]
        expected_numbers = [
            float(x)
            for x in expected.split()
            if x.replace(".", "").isdigit()
            or (x.startswith("-") and x[1:].replace(".", "").isdigit())
        ]

        # Simple accuracy check
        if len(response_numbers) > 0 and len(expected_numbers) > 0:
            accuracy = (
                1.0 if abs(response_numbers[0] - expected_numbers[0]) < 0.01 else 0.0
            )
        else:
            accuracy = 0.0

        return {
            "accuracy": accuracy,
            "response_numbers": response_numbers,
            "expected_numbers": expected_numbers,
        }
    except Exception as e:
        return {"accuracy": 0.0, "error": str(e)}


@weave.op()
def get_calculation_evaluation(expected: str, output: dict) -> dict:
    # Here is where you'd define the logic to score the model output
    result = calculate_accuracy_evaluation(
        response=output["generated_text"], expected=expected
    )
    return {"match": result["accuracy"] == 1.0}


@weave.op()
def evaluate_multiple_calculations(
    results: List[Dict[str, Any]], expected_answers: List[str]
) -> Dict[str, Any]:
    """Evaluate multiple calculation results"""
    if len(results) != len(expected_answers):
        raise ValueError("Number of results must match number of expected answers")

    total_accuracy = 0
    evaluation_results = []

    for i, (result, expected) in enumerate(zip(results, expected_answers)):
        response = result.get("answer", "")
        evaluation = calculate_accuracy_evaluation(response, expected)
        evaluation_results.append(
            {
                "question": result.get("question", f"Question {i+1}"),
                "expected": expected,
                "actual": response,
                "evaluation": evaluation,
            }
        )
        total_accuracy += evaluation["accuracy"]

    average_accuracy = total_accuracy / len(results) if results else 0

    return {
        "average_accuracy": average_accuracy,
        "total_questions": len(results),
        "individual_evaluations": evaluation_results,
        "successful_evaluations": len(
            [r for r in evaluation_results if r["evaluation"]["accuracy"] > 0]
        ),
    }

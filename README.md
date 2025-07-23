# Strands Evaluation

## Basic Usage

```python
from strands import Agent
from strands_evaluation.dataset import Dataset
from strands_evaluation.case import Case
from strands_evaluation.evaluators.llm_evaluator import LLMEvaluator

# 1. Create test cases
test_cases = [
    Case[str, str](
        name="knowledge-1",
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        metadata={"category": "knowledge"}
    ),
    Case[str, str](
        name="knowledge-2",
        input="What color is the ocean?",
        metadata={"category": "knowledge"}
    )
]

# 2. Create an evaluator
evaluator = LLMEvaluator(
    rubric="The output should represent a reasonable answer to the input."
)

# 3. Create a dataset
dataset = Dataset[str, str](
    cases=test_cases,
    evaluator=evaluator
)

# 4. Define a task function
def get_response(query: str) -> str:
    agent = Agent(callback_handler=None)
    return str(agent(query))

# 5. Run evaluations
report = dataset.run_evaluations(get_response)
report.display()
```

## Saving and Loading Datasets

```python
# Save dataset to JSON
dataset.to_file("my_dataset", "json")

# Load dataset from JSON
loaded_dataset = Dataset.from_file("./dataset_files/my_dataset.json", "json")
```

## Custom Evaluators

```python
from strands_evaluation.evaluators.evaluator import Evaluator
from strands_evaluation.types.evaluation import EvaluationData, EvaluationOutput

class CustomEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Custom evaluation logic
        if evaluation_case.expected_output in evaluation_case.actual_output:
            score = 1.0
            test_pass = True
        else:
            score = 0.0
            test_pass = False
            
        return EvaluationOutput(
            score=score,
            test_pass=test_pass,
            reason="Custom evaluation reason"
        )

# Use custom evaluator
dataset = Dataset[str, str](
    cases=test_cases,
    evaluator=CustomEvaluator()
)
```

## Evaluating Tool Usage

```python
from strands_tools import calculator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator

# Create test cases with expected tool trajectories
test_case = Case[str, str](
    name="calculator-1",
    input="What is the square root of 9?",
    expected_output="The square root of 9 is 3.",
    expected_trajectory=["calculator"],
    metadata={"category": "math"}
)

# Create trajectory evaluator
trajectory_evaluator = TrajectoryEvaluator(
    rubric="The trajectory should represent a reasonable use of tools based on the input.",
    include_inputs=True
)

# Define task that returns tool usage
def get_response_with_tools(query: str) -> dict:
    agent = Agent(tools=[calculator])
    response = agent(query)
    
    return {
        "output": str(response),
        "trajectory": list(response.metrics.tool_metrics.keys())
    }

# Create dataset and run evaluations
dataset = Dataset[str, str](
    cases=[test_case],
    evaluator=trajectory_evaluator
)

report = dataset.run_evaluations(get_response_with_tools)
```

## More Examples

See the `examples` directory for more detailed examples.

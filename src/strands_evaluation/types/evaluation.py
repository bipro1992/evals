from pydantic import BaseModel
from typing_extensions import Any, Generic, TypedDict, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(TypedDict, total=False):
    """
    Represents a single interaction in a multi-agent or multi-step system.

    Used to capture the communication flow and dependencies between different
    components (agents, tools, or processing nodes) during task execution.
    All fields are optional to accommodate different interaction patterns.

    Attributes:
        node_name: Identifier for the agent, tool, or component involved in this interaction
        dependencies: List of other nodes/components this interaction depends on or references
        messages: Sequence of messages, responses, or communication exchanged during this interaction

    Example:
        interaction = {
            "node_name": "calculator_agent",
            "dependencies": ["input_parser", "math_validator"],
            "messages": ["Calculate 2+2"]
        }
    """

    node_name: str
    dependencies: list[str] | None
    messages: list[str] | None


class EvaluationData(BaseModel, Generic[InputT, OutputT]):
    """
    A record of all of the context for the evaluator to evaluate a test case.

    Attributes:
        input: The input to the task. eg. the query to the agent
        actual_output: The actual response given the input.
        expected_output: The expected response given the input.
        actual_trajectory: The actual trajectory of a task given the input.
        expected_trajectory: The expected trajectory of a task given the input.
        name: The name of the test case. This will be used to identify the test in the summary report.
        metadata: Additional information about the test case.
        actual_interactions: The actual interaction sequence given the input.
        expected_interactions: The expected interaction sequence given the input.
    """

    input: InputT
    actual_output: OutputT | None = None
    name: str | None = None
    expected_output: OutputT | None = None
    expected_trajectory: list[Any] | None = None
    actual_trajectory: list[Any] | None = None
    metadata: dict[str, Any] | None = None
    actual_interactions: list[Interaction] | None = None
    expected_interactions: list[Interaction] | None = None


class EvaluationOutput(BaseModel):
    """
    Structured output for LLM-based judge.

    Attributes:
        score: The score of the test case.
        test_pass: Whether the test pass or fail.
        reason: The reason for the score for each test case.
    """

    score: float
    test_pass: bool
    reason: str | None = None

from typing import Generic
from typing_extensions import TypeVar
from dataclasses import dataclass

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

from ..types.evaluation import EvaluationData, EvaluationOutput

@dataclass
class Evaluator(Generic[InputT, OutputT]):
    """
    Base class for evaluators.

    Evaluators can assess the performance of a task on all test cases.
    Subclasses must implement the `evaluate` method.
    """
    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
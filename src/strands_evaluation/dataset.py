from pydantic import BaseModel
from typing import Dict, Any, Generic, Callable
from typing_extensions import TypeVar

from .case import Case
from .types.evaluation import EvaluationData, EvaluationReport
from .evaluators.evaluator import Evaluator

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class Dataset(BaseModel, Generic[InputT, OutputT]):
    """
    A collection of test cases, representing a dataset.

    Dataset organizes a collection of test cases and evaluate them all with
    the defined evaluator on some task. 

    Attributes:
        cases: A list of test cases in the dataset.
        evaluator: The evaluator to be used on the test cases.

    Example:
        dataset = Dataset[str, str](
            cases=[
                Case(name="Simple Knowledge",
                        input="What is the capital of France?",
                        expected_output="The capital of France is Paris.",
                        expected_trajectory=[],
                        metadata={"category": "knowledge"}),
               Case(name="Simple Math",
                        input="What is 2x2?",
                        expected_output="2x2 is 4.",
                        expected_trajectory=["calculator],
                        metadata={"category": "math"})
            ],
            evaluator=Evaluator()
        )
    """
    cases: list[Case[InputT, OutputT]]
    evaluator: Evaluator[InputT, OutputT]

    def _run_task(self, task: Callable[[InputT], OutputT | Dict[str, Any]], case: Case[InputT, OutputT]) -> EvaluationData[InputT, OutputT]:       
        """
        Run the task with the inputs from the test case.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
            case: The test case containing neccessary information to run the task

        Return:
            An EvaluationData record containing the input and actual output, name, expected output, and metadata.
        """
        evaluation_context = EvaluationData(name=case.name,
                                            input = case.input,
                                            expected_output=case.expected_output,
                                            expected_trajectory=case.expected_trajectory,
                                            metadata=case.metadata)
        task_output = task(case.input)
        if isinstance(task_output, dict): # could be evaluating the trajectory as well
            evaluation_context.actual_output = task_output.get("output")
            evaluation_context.actual_trajectory = task_output.get("trajectory")
        else: # evaluating only the output
            evaluation_context.actual_output = task_output
        return evaluation_context

    def run_evaluations(self, task: Callable[[InputT], OutputT | Dict[str, Any]]) -> EvaluationReport:
        """
        Run the evaluations for all of the test cases with the evaluator.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
                                     
        Return:
            An EvaluationReport containing the overall score, individual case results, and basic feedback for each test case.
        """        
        scores = []
        test_passes = []
        cases = []         
        reasons = []              
        for case in self.cases:
            try:
                evaluation_context = self._run_task(task, case)
                evaluation_output = self.evaluator.evaluate(evaluation_context)
                cases.append(evaluation_context.model_dump())
                test_passes.append(evaluation_output.test_pass)
                scores.append(evaluation_output.score)
                if evaluation_output.reason:
                    reasons.append(evaluation_output.reason)
                else:
                    reasons.append("")
            except Exception as e:
                cases.append(case.model_dump())
                test_passes.append(False)
                scores.append(0)
                reasons.append(f"An error occured : {str(e)}")

        report = EvaluationReport(overall_score = sum(scores)/len(scores) if len(scores) else 0,
                                  scores = scores,
                                  test_passes = test_passes,
                                  cases = cases,
                                  reasons=reasons)

        return report
    

if __name__ == "__main__":
  pass
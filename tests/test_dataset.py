import pytest
from src.strands_evaluation.dataset import Dataset
from src.strands_evaluation.case import Case
from src.strands_evaluation.evaluators.evaluator import Evaluator
from src.strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData


class MockEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Simple mock: pass if actual equals expected
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Mock evaluation")

@pytest.fixture
def mock_evaluator():
    return MockEvaluator()

class TestDataset:
    
    def test_create_dataset(self, mock_evaluator):
        """Test creating a Dataset with test cases and evaluator"""
        cases = [
            Case(name="test1", input="hello", expected_output="world"),
            Case(name="test2", input="foo", expected_output="bar")
        ]
        
        dataset = Dataset(cases=cases, evaluator=mock_evaluator)
        
        assert len(dataset.cases) == 2
        assert dataset.evaluator == mock_evaluator
    
    def test_run_task_simple_output(self, mock_evaluator):
        """Test _run_task with simple output"""
        case = Case(name="test", input="hello", expected_output="world")
        dataset = Dataset(cases=[case], evaluator=mock_evaluator)
        
        def simple_task(input_val):
            return f"response to {input_val}"
        
        result = dataset._run_task(simple_task, case)
        
        assert result.input == "hello"
        assert result.actual_output == "response to hello"
        assert result.expected_output == "world"
        assert result.name == "test"
        assert result.expected_trajectory is None
        assert result.actual_trajectory is None
        assert result.metadata == {}
    
    def test_run_task_dict_output(self, mock_evaluator):
        """Test _run_task with dictionary output containing trajectory"""
        case = Case(name="test", input="hello", expected_output="world")
        dataset = Dataset(cases=[case], evaluator=mock_evaluator)
        
        def dict_task(input_val):
            return {
                "output": f"response to {input_val}",
                "trajectory": ["step1", "step2"]
            }
        
        result = dataset._run_task(dict_task, case)
        
        assert result.actual_output == "response to hello"
        assert result.actual_trajectory == ["step1", "step2"]
    
    def test_run_evaluations(self, mock_evaluator):
        """Test complete evaluation run"""
        cases = [
            Case(name="match", input="hello", expected_output="hello"),
            Case(name="no_match", input="foo", expected_output="bar")
        ]
        dataset = Dataset(cases=cases, evaluator=mock_evaluator)
        
        def echo_task(input_val):
            return input_val
        
        report = dataset.run_evaluations(echo_task)
        
        assert len(report.scores) == 2
        assert report.scores[0] == 1.0  # match
        assert report.scores[1] == 0.0  # no match
        assert report.test_passes[0] == True
        assert report.test_passes[1] == False
        assert report.overall_score == 0.5
        assert len(report.cases) == 2
    
    def test_empty_dataset(self, mock_evaluator):
        """Test dataset with no test cases"""
        dataset = Dataset(cases=[], evaluator=mock_evaluator)
        
        def dummy_task(input_val):
            return input_val
        
       
        report = dataset.run_evaluations(dummy_task)
        assert report.overall_score == 0
        assert len(report.scores) == 0
        assert len(report.cases) == 0
        assert len(report.test_passes) == 0
        assert len(report.reasons) == 0

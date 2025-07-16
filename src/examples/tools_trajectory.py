from strands import Agent
from strands_tools import calculator

from strands_evaluation.dataset import Dataset 
from strands_evaluation.case import Case
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator

def tools_trajectory_example():
    ### Step 1: Create test cases ###
    test_case1 = Case[str, str](name = "calculator-1",
                                    input="What is the square root of 9?",
                                    expected_output="The square root of 9 is 3.",
                                    expected_trajectory=["calculator"],
                                    metadata={"category": "math"})
                                    
    test_case2 = Case[str, str](name = "calculator-2",
                                    input="What is 2x2?",
                                    expected_output="4",
                                    metadata={"category": "math"})
                                    
    
    ### Step 2: Create evaluator ###   
    trajectory_evaluator = TrajectoryEvaluator(rubric = "The trajectory should represent a reasonable use of tools based on the input.",
                                            include_inputs = True)

    ### Step 3: Create dataset ###     
    dataset = Dataset[str, str](cases = [test_case1, test_case2], evaluator = trajectory_evaluator)

    ### Step 4: Define task ###  
    def get_response(query: str) -> dict:
        agent = Agent(tools = [calculator])
        response = agent(query)
        print({"output": str(response),
                "trajectory": list(response.metrics.tool_metrics.keys())})
        
        return {"output": str(response),
                "trajectory": list(response.metrics.tool_metrics.keys())}
    
    ### Step 5: Run evaluation ###                                                                                                                                                
    report = dataset.run_evaluations(get_response)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.tools_trajectory 
    report = tools_trajectory_example()
    # print(report)
    report.display()



from strands import Agent
from strands_tools import calculator

from strands_evaluation.dataset import Dataset 
from strands_evaluation.case import Case
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator

def messages_trajectory():
    ### Step 1: Create test cases ###
    test_case1 = Case[list, str](name = "intro",
                                    input=["Hello", "What tools do you have?", "How can I use them?"])
                                    
    test_case2 = Case[list, str](name = "math teacher",
                                    input=["How do I take the derivatives?",
                                        "How do I take the derivative of 2x with respect to x?"])
                                    
    
    ### Step 2: Create evaluator ###   
    trajectory_evaluator = TrajectoryEvaluator(rubric = "The trajectory should represent a reasonable sequence of messages based on the inputs.",
                                            include_inputs = True)

    ### Step 3: Create dataset ###     
    dataset = Dataset[list, str](cases = [test_case1, test_case2], evaluator = trajectory_evaluator)

    ### Step 4: Define task ###  
    def get_response(inputs: list) -> dict:
        agent = Agent(tools = [calculator], callback_handler=None)
        for query in inputs:
            response = agent(query)

        return {"trajectory": list(agent.messages)}
    
    ### Step 5: Run evaluation ###                                                                                                                                                
    report = dataset.run_evaluations(get_response)
    return report


def memory_messages_trajectory():
    ### Step 1: Create test cases ###
    test_case1 = Case[list, str](name = "intro",
                                    expected_output="""Looking at our conversation history, you've asked me:
                                                    1. **"Hello"** - You greeted me first
                                                    2. **"What tools do you have?"** - You asked about my available tools (which I just provided a comprehensive list for)
                                                    That's our complete conversation so far!""",
                                    input=["Hello", "What tools do you have?", "What did I asked you before?"])
                                    
    ### Step 2: Create evaluator ###   
    trajectory_evaluator = TrajectoryEvaluator(rubric = "The trajectory should represent a reasonable sequence of messages based on the inputs.",
                                            include_inputs = True)

    ### Step 3: Create dataset ###     
    dataset = Dataset[list, str](cases = [test_case1], evaluator = trajectory_evaluator)

    ### Step 4: Define task ###  
    def get_response(inputs: list) -> dict:
        agent = Agent(tools = [calculator], callback_handler=None)
        response = None
        for query in inputs:
            response = agent(query)

        return {
            "output": str(response),
            "trajectory": list(agent.messages)}
    
    ### Step 5: Run evaluation ###                                                                                                                                                
    report = dataset.run_evaluations(get_response)
    return report

if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.messages_trajectory 
    report = messages_trajectory()
    # # print(report)
    report.display(include_output=False)

    report = memory_messages_trajectory()
    report.display()



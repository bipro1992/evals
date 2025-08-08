import asyncio

from strands import Agent
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.interactions_evaluator import InteractionsEvaluator
from strands_evaluation.evaluators.output_evaluator import OutputEvaluator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator
from strands_evaluation.evaluators.utils import helper_funcs
from strands_evaluation.generators.dataset_generator import DatasetGenerator
from typing_extensions import TypedDict


async def test_dataset_generator():
    class TrajectoryType(TypedDict):
        tool: str
        input: dict

    generator = DatasetGenerator[str, float](
        str, float, trajectory_type=TrajectoryType, include_expected_trajectory=True
    )
    dataset = await generator.from_context_async(
        "Create test cases about math given that you have access to these tools: calculator, python.",
        task_description="Getting the response from an AI agent with access to tools.",
        num_cases=10,
        evaluator=TrajectoryEvaluator,
    )
    print(len(dataset.cases))
    dataset.to_file("generated_traj_dataset_context")

    generator = DatasetGenerator[str, str](str, str, trajectory_type=TrajectoryType, include_expected_interactions=True)
    dataset = await generator.from_context_async(
        "Create test cases about research for a multi-agent system with the following agents: researcher, analyst, fact_checker, and report_writer.",
        task_description="Getting the response from an AI agent with access to tools.",
        num_cases=5,
        evaluator=InteractionsEvaluator,
    )
    print(len(dataset.cases))
    dataset.to_file("generated_interaction_dataset_context")

    generator = DatasetGenerator[str, str](str, str, trajectory_type=TrajectoryType, include_expected_interactions=True)
    dataset = await generator.from_scratch_async(
        ["math", "science"],
        task_description="Getting the response from an AI agent.",
        num_cases=5,
        evaluator=OutputEvaluator,
    )
    dataset.to_file("generated_output_dataset_scratch")
    print(len(dataset.cases))

    # try from_dataset
    generator = DatasetGenerator[str, str](str, str, trajectory_type=TrajectoryType, include_expected_interactions=True)
    dataset = Dataset.from_file("dataset_files/generated_interaction_dataset_context.json")
    new_dataset = await generator.update_current_dataset_async(
        dataset,
        "Getting the response from an AI agent with access to tools.",
        context="Create test cases about research for a multi-agent system with the following agents: researcher, analyst, fact_checker, and report_writer.",
        num_cases=5,
        new_evaluator_type=OutputEvaluator,
    )
    new_dataset.to_file("generated_output_dataset_from_dataset")
    print(len(new_dataset.cases))

    # new from dataset
    generator = DatasetGenerator[str, str](str, str, trajectory_type=TrajectoryType, include_expected_trajectory=True)
    dataset = Dataset.from_file("dataset_files/generated_traj_dataset_context.json")
    new_dataset = await generator.from_dataset_async(
        dataset,
        "Getting the response from an AI agent with access to tools.",
        extra_information="Create test cases about math given that you have access to these tools: calculator, python.",
    )

    new_dataset.to_file("generated_traj_dataset_from_dataset")
    print(len(new_dataset.cases))

    # try evaluating them
    dataset = Dataset.from_file("dataset_files/generated_traj_dataset_context.json")

    def task_func(input: str) -> dict:
        agent = Agent(system_prompt="You are a helpful assistant that can do math.", callback_handler=None)
        output = agent(input)
        return {
            "output": str(output),
            "trajectory": helper_funcs.extract_agent_tools_used_from_messages(agent.messages),
        }

    report = await dataset.run_evaluations_async(task_func)
    report.run_display()


if __name__ == "__main__":
    # python -m examples.try_dataset_generator
    asyncio.run(test_dataset_generator())

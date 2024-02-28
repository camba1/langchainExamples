# Use a Plan and Execute type of Agent to run a math solving tool
from langchain.chains import LLMMathChain
from langchain_openai import ChatOpenAI
from langchain.agents.tools import Tool
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner


def solve_math_problem(query, agent_verbose_output=False):
    """
    Use an agent to solve the math problem in the query. At the time of this writing, plan and execute chains
    are still experimental. Note that sometimes the function will fail thinking that it needs to pass a string
    to numexpr, but it works in most cases
    :param agent_verbose_output: Indicates if the agent will return additional information or not
    :param query: Math problem to be solved
    :return:
    """

    #  Create a math chain
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0, model=llm_model)
    llm_math = LLMMathChain.from_llm(llm, verbose=True)

    # define our tool
    calculator_tool = Tool(
        name="calculator",
        func=llm_math.run,
        description="useful for when you need to answer questions about math",
    )

    # Define our agent
    tools = [calculator_tool]
    planner = load_chat_planner(llm)
    print(planner.llm_chain.prompt.messages[0].content)
    executor = load_agent_executor(llm, tools, verbose=agent_verbose_output)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=agent_verbose_output)
    print(executor.chain.agent.llm_chain.prompt.messages[0].prompt.template)

    # Execute agent to solve math problem
    print(agent.invoke(query))


def app_run():
    solve_math_problem({"input":"What is the 25% of 300?"}, agent_verbose_output=False)


app_run()
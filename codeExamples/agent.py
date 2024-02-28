from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI


def solve_math_problem(query, agent_verbose_output=False):
    """
    Use a basic agent to solve the math problem in the query.
    :param agent_verbose_output: Indicates if the agent will return additional information or not
    :param query: Math problem to be solved
    :return:
    """
    llm_model = "gpt-3.5-turbo"

    llm = ChatOpenAI(temperature=0, model=llm_model)

    tools = load_tools(["llm-math"], llm=llm)

    agent = initialize_agent(tools, llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             handle_parsing_errors=True,   # Agent to try to handle response that is not correctly formatted
                             verbose=agent_verbose_output)

    print(agent.invoke(query))

def app_run():
    solve_math_problem({"input": "What is the 25% of 300?"}, agent_verbose_output=False)

app_run()
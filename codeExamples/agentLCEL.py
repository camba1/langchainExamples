# Create a ReAct type Agent to solve a math problem written with LCEL
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools

from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor


def solve_math_problem(query, agent_verbose_output=False):
    """
    Use an agent setup with LCEL to solve the math problem in the query.
    :param agent_verbose_output: Indicates if the agent will return additional information or not
    :param query: Math problem to be solved
    :return:
    """
    # setup LLM
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0, model=llm_model)

    # Define our list of tools
    tools = load_tools(['llm-math'], llm=llm)

    # Build our prompt
    prompt = hub.pull("hwchase17/react")  # Pull pre-existing prompt from langchain hub

    print("---------ORIGINAL PROMPT-------------")
    print(prompt)
    print("---------RENDER TEXT DESC-------------")
    print(render_text_description(tools))

    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    print("---------MODIFIED PROMPT-------------")
    print(prompt)

    # Line below to avoid error LLM output:
    # "Parsing LLM output produced both a final answer and a parse-able action:"
    llm_with_stop = llm.bind(stop=["\nObservation"])

    #  Define and execute agent
    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_stop
            | ReActSingleInputOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=agent_verbose_output)

    print("---------INVOKE EXECUTOR-------------")
    result = agent_executor.invoke(query)

    print(result)


def app_run():
    solve_math_problem({"input": "What is the 25% of 300?"}, agent_verbose_output=False)


app_run()

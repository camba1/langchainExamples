# Create a ReAct type Agent to solve a math problem written with LCEL
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools

from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0, model=llm_model)

tools = load_tools(['llm-math'], llm=llm)

prompt = hub.pull("hwchase17/react")   # Pull pre-existing prompt from langchain hub

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

llm_with_stop = llm.bind(stop=["\nObservation"])  # To avoid error LLM output: "Parsing LLM output produced both a final answer and a parse-able action:"

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("---------INVOKE EXECUTOR-------------")
result = agent_executor.invoke({"input": "What is the 25% of 300?"})

print(result)
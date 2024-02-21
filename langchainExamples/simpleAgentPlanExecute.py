# Use a Plan and Execute type of Agent to run a math solving tool
from langchain.chains import LLMMathChain
from langchain_openai import ChatOpenAI
from langchain.agents.tools import Tool
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0, model=llm_model)
llm_math = LLMMathChain.from_llm(llm, verbose=True)


calculator_tool = Tool(
    name="calculator",
    func=llm_math.run,
    description="useful for when you need to answer questions about math",
)

tools = [calculator_tool]
planner = load_chat_planner(llm)
print(planner.llm_chain.prompt.messages[0].content)

executor = load_agent_executor(llm, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

print(executor.chain.agent.llm_chain.prompt.messages[0].prompt.template)


print(agent.invoke("What is the 25% of 300?"))


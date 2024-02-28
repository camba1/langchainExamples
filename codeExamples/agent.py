from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0, model=llm_model)

tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(tools, llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         handle_parsing_errors=True,   # tell agent to try to handle response that is not correclty formatted
                         verbose=True)

print(agent.invoke("What is the 25% of 300?"))
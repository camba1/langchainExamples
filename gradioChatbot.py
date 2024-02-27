import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

AI_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.9, openai_api_key=AI_KEY, model_name="gpt-3.5-turbo")


def predict(prompt, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=prompt))
    gpt_response = llm.stream(history_langchain_format)
    # gpt_response = llm(history_langchain_format)
    # return gpt_response.content
    partial_message = ""
    for chunk in gpt_response:
        partial_message = partial_message + chunk.dict()['content']
        yield partial_message

gr.ChatInterface(predict).launch()

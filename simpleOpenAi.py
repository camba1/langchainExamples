import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


llm_model = "gpt-3.5-turbo"

def define_prompt():

    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse,\
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """

    style = """American English \
    in a calm and respectful tone
    """

    prompt = f"""Translate the text \
    that is delimited by triple backticks 
    into a style that is {style}.
    text: ```{customer_email}```
    """

    return prompt

def get_completion(llmprompt, model=llm_model):
    messages = [{"role": "user", "content": llmprompt}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=0)
    return response.choices[0].message.content



print(get_completion(define_prompt()))

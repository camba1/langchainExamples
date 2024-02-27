# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm_model = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.0, model=llm_model)


def build_prompt_template():
    template_string = """Translate the text \
    that is delimited by triple backticks \
    into a style that is {style}. \
    text: ```{text}```
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    return prompt_template


def build_customer_chat_prompt(prompt_template):
    print(prompt_template.messages[0].prompt)
    print(prompt_template.messages[0].prompt.input_variables)

    customer_style = """American English \
    in a calm and respectful tone
    """

    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse, \
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """

    customer_messages = prompt_template.format_messages(
        style=customer_style,
        text=customer_email)

    print(type(customer_messages))
    print(type(customer_messages[0]))

    print(customer_messages[0])

    # Call the LLM to translate to the style of the customer message
    customer_response = chat.invoke(customer_messages)

    print(customer_response.content)


def build_service_chat_prompt(prompt_template):
    service_reply = """Hey there customer, \
    the warranty does not cover \
    cleaning expenses for your kitchen \
    because it's your fault that \
    you misused your blender \
    by forgetting to put the lid on before \
    starting the blender. \
    Tough luck! See ya!
    """

    service_style_pirate = """\
    a polite tone \
    that speaks in English Pirate\
    """

    service_messages = prompt_template.format_messages(
        style=service_style_pirate,
        text=service_reply)

    print(service_messages[0].content)

    service_response = chat.invoke(service_messages)
    print(service_response.content)


prompt_template = build_prompt_template()
build_customer_chat_prompt(prompt_template)
build_service_chat_prompt(prompt_template)

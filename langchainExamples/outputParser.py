from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


def create_chat_templates(customer_review, output_format_instructions):
    review_template = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    text: {text}
    
    {format_instructions}
    """

    prompt_template = ChatPromptTemplate.from_template(review_template)
    print(prompt_template)

    messages = prompt_template.format_messages(text=customer_review,
                                               format_instructions=output_format_instructions)
    return messages


def execute_chat(messages):
    llm_model = "gpt-3.5-turbo"
    chat = ChatOpenAI(temperature=0.0, model=llm_model)

    response = chat.invoke(messages)
    print(response.content)

    print(type(response.content))
    return response


def define_output_parser():
    gift_schema = ResponseSchema(name="gift",
                                 description="Was the item purchased\
                                 as a gift for someone else? \
                                 Answer True if yes,\
                                 False if not or unknown.")
    delivery_days_schema = ResponseSchema(name="delivery_days",
                                          description="How many days\
                                          did it take for the product\
                                          to arrive? If this \
                                          information is not found,\
                                          output -1.")
    price_value_schema = ResponseSchema(name="price_value",
                                        description="Extract any\
                                        sentences about the value or \
                                        price, and output them as a \
                                        comma separated Python list.")
    response_schemas = [gift_schema,
                        delivery_days_schema,
                        price_value_schema]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser


def format_output(response, output_parser):
    format_instructions = output_parser.get_format_instructions()
    print(format_instructions)
    output_dict = output_parser.parse(response.content)
    print(output_dict)
    print(type(output_dict))
    return output_dict


def run_app():
    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """

    output_parser = define_output_parser()
    messages = create_chat_templates(customer_review, output_parser.get_format_instructions())
    response = execute_chat(messages)
    output_dict = format_output(response, output_parser)
    print(output_dict.get('delivery_days'))


run_app()

# {
#   "gift": False,
#   "delivery_days": 5,
#   "price_value": "pretty affordable!"
# }

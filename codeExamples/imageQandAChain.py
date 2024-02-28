from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

import base64

model = "gpt-4-vision-preview"
llm = ChatOpenAI(model=model, max_tokens=256)


def question_image_from_url(url, detail, question):
    """
    Ask a question about the content of an image pulled from a URL
    :param question:Question to be ask to the LLM
    :param url: URL pointing to the image
    :param detail: Amount of detail the LLM should use. Either auto/high/low
    """
    human_msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": detail, },
        },
    ])
    result = llm.invoke([human_msg])
    print(result)


def encode_image(image_path):
    """
    Encode an image to base64
    :param image_path: Path to the image file
    :return: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


def question_image_from_file(image_path, detail, question):
    """
    Ask a question about the content of an image pulled from a URL
    :param question:Question to be ask to the LLM
    :param image_path: filepath pointing to the image file
    :param detail: Amount of detail the LLM should use. Either auto/high/low
    """
    encoded_image = encode_image(image_path)
    human_msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}",
                "detail": detail, },
        },
    ])

    result = llm.invoke([human_msg])
    print(result)


def app_run():
    """
    Ask a question from an image in a URL and the ask a question about a local image uploaded to Open AI
    """

    detail = "auto"
    question = "what is this image showing. Be concise."
    image_url = "https://raw.githubusercontent.com/camba1/gotemp/master/diagramsforDocs/UI_Promo_detail.png"
    question_image_from_url(image_url, detail, question)

    image_path = "data/UI_Promo_detail.png"
    question = "what is the validity date of the promotion in this image"
    question_image_from_file(image_path, detail, question)


app_run()

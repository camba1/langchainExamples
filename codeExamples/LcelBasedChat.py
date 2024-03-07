from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# RAG imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain import hub
from langchain.schema.runnable import RunnableMap

# async imports
import asyncio

model_name = "gpt-3.5-turbo"
model = ChatOpenAI(temperature=0.5, model_name=model_name)
output_parser = StrOutputParser()


def simple_chain():
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | model | output_parser
    result = chain.invoke({"topic": "peanuts"})
    print(f"Simple chain result: {result}")


def simple_rag_chain():
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prompt = hub.pull("rlm/rag-prompt")
    # print(f"RAG prompt: {prompt}")

    vector_store = Qdrant.from_texts(texts=["Harrison played in a band in Chicago", "bears like to win super bowls"],
                                     embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                                     location=":memory:",
                                     collection_name="rag_collection"
                                     )
    retriever = vector_store.as_retriever()

    # print(f"Similarity search result: {retriever.get_relevant_documents("Where did Harrison play")}")

    prompt_param_mapper = RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    })

    # run the mapper and print the results to see the values that will be passed to the prompt
    print(f"RAG prompt param mapper: {prompt_param_mapper.invoke({"question": "Where did Harrison play"})}")

    chain = prompt_param_mapper | prompt | model | output_parser
    result = chain.invoke({"question": "Where did Harrison play "})

    print(f"RAG chain result: {result}")


def bind_openai_functions():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}")
        ]
    )
    # For our purposes, we are not creating the two functions named in this function object
    # since we are just going to pass this to OpenAI, and it just returns the function name to use
    #  along with its arguments. Which function it returns depends on the prompt we give it (if any)
    functions = [
        {
            "name": "weather_search",
            "description": "Search for weather given an airport code",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "The airport code to get the weather for"
                    },
                },
                "required": ["airport_code"]
            }
        },
        {
            "name": "sports_search",
            "description": "Search for news of recent sport events",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "The sports team to search for"
                    },
                },
                "required": ["team_name"]
            }
        }
    ]

    model2 = ChatOpenAI(temperature=0, model_name=model_name)
    model2 = model2.bind_functions(functions)
    chain = prompt | model2
    result = chain.invoke({"input": "how did the bears do yesterday?"})
    print(f"Bind functions result 1: {result}")
    result = chain.invoke({"input": "what is the weather in ord?"})
    print(f"Bind functions result 2: {result}")
    result = chain.invoke({"input": "who are you?"})
    print(f"Bind functions result 3: {result}")


def chain_with_fallback():
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a nice assistant who always includes a compliment in your response",
            ),
            ("human", "Why did the {animal} cross the road"),
        ]
    )

    chat_model = ChatOpenAI(model_name="gpt-fake")
    bad_chain = chat_prompt | chat_model | StrOutputParser()

    # This line fails because there is no gpt-fake model
    # bad_chain.invoke({"animal": "chicken"})

    good_chain = chat_prompt | model | StrOutputParser()

    # This succeeds because it calls the bad model first and sice tha fails, it automatically
    # calls the good model.
    chain = bad_chain.with_fallbacks([good_chain])
    result = chain.invoke({"animal": "chicken"})
    print(result)


async def batch_stream_asynch():
    prompt = ChatPromptTemplate.from_template(
        "Tell me a short joke about {topic}"
    )

    chain = prompt | model | output_parser

    # Invoke normally
    result = chain.invoke({"topic": "turtles"})
    print(result)

    # Invoke multiple times (batch) in parallel (as much as possible)
    result = chain.batch([{"topic": "birds"}, {"topic": "elephants"}])
    print(result)

    # Invoke as stream
    for t in chain.stream({"topic": "cats"}):
        print(t)

    # invoke asynch
    result = await chain.ainvoke({"topic": "dogs"})
    print(result)


def app_run():
    # simple_chain()
    # simple_rag_chain()
    # bind_openai_functions()
    # chain_with_fallback()
    asyncio.run(batch_stream_asynch())


app_run()

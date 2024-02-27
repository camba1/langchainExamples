import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Import for simple sequential chains
from langchain.chains import SimpleSequentialChain

# Import for sequential chains
from langchain.chains import SequentialChain

# Imports for router chains
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
# End imports for router chains

llm_model = "gpt-3.5-turbo"


def load_data(filepath):
    """
    Load a csv file into a pandas dataframe.
    :param filepath: path to the local file to load
    :return: The dataframe head
    """
    df = pd.read_csv(filepath)
    print(df.head())
    return df.head()


def basic_llm_chain(product, llm):
    """
    Runs a basic chain to answer a  question about a product
    :param product: The product for which we need information
    :param llm: Model to use in the chat
    """
    prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}")
    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.invoke(product)
    print(result)


def simple_sequential(product, llm):
    """
    Simple two-step sequential chain where we first determine who makes a particular product, and then we
    write a quick summary about the manufacturer. Each chain in the sequence can take only one input and one output
    :param product:
    :param llm: Model to be used
    """

    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
        a company that makes {product}?"
    )

    # Chain 1
    chain_one = LLMChain(llm=llm, prompt=first_prompt)

    # prompt template 2
    second_prompt = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following \
        company:{company_name}"
    )
    # chain 2
    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    # Combine both chains pieces into one big chain
    overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                                 verbose=True
                                                 )
    result = overall_simple_chain.invoke(product)
    print(result)


def sequential_chain(df_product_reviews, review_index, llm):
    """
    Simple two-step sequential chain where we first determine who makes a particular product, and then we
    write a quick summary about the manufacturer. Each chain in the sequence can take
     multiple one inputs and one outputs
    :param review_index: The index of the review to use in the sequential chain
    :param df_product_reviews: information about the reviews of the different products
    :param llm: Model to be used
    """
    # prompt template 1: translate to english
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to english:"
        "\n\n{Review}"
    )
    # chain 1: input= Review and output= English_Review
    chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:"
        "\n\n{English_Review}"
    )
    # chain 2: input= English_Review and output= summary
    chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

    # prompt template 3: translate to english
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    # chain 3: input= Review and output= language
    chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

    # prompt template 4: follow-up message
    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    # chain 4: input= summary, language and output= followup_message
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

    # overall_chain: input= Review
    # and output= English_Review,summary, followup_message
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three, chain_four],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "followup_message"],
        verbose=True
    )

    review = df_product_reviews.Review[review_index]
    print(review)
    result = overall_chain.invoke(review)
    print(result)


def router_chain_templates():
    """
    Build the different prompt templates to be used in the router chain
    :return: list of prompt templates
    """
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""

    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""

    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    templates = {"physics_template": physics_template,
                 "math_template": math_template,
                 "history_template": history_template,
                 "computerscience_template": computerscience_template}

    return templates


def router_chain_prompt_infos(templates):
    """
    Build the list containing the JSON objects that contain sll the information needed by the router chain
    to use the different prompt templates we are providing
    :param templates: The list of prompt templates
    :return: List of JSON prompt templates in a format that is usable by the router chain
    """
    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": templates["physics_template"]
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": templates["math_template"]
        },
        {
            "name": "History",
            "description": "Good for answering history questions",
            "prompt_template": templates["history_template"]
        },
        {
            "name": "computer science",
            "description": "Good for answering computer science questions",
            "prompt_template": templates["computerscience_template"]
        }
    ]
    return prompt_infos


def build_router_template_prompt():
    """
    Build the overall prompt to be used by the router chain, it contains, among ohter things, place holders
    for our prompt templates ("candidate prompts)
    :return: final prompt template string
    """
    router_prompt_template = """Given a raw text input to a 
    language model select the model prompt best suited for the input. 
    You will be given the names of the available prompts and a 
    description of what the prompt is best suited for. 
    You may also revise the original input if you think that revising
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \\ name of the prompt to use or "DEFAULT"
        "next_inputs": string \\ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt 
    names specified below OR it can be "DEFAULT" if the input is not
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input 
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

    return router_prompt_template


def build_router_chain(prompt_infos, router_prompt_template, llm):
    """
    Build the router chain as a combination of multiple individual chains
    :param prompt_infos: The diferent prompt templates that can be used by the chain
    :param router_prompt_template: Final Chain prompt template string
    :param llm: Model to be used
    :return: Chain object
    """
    # loop through prompt_infos and create a chain for each prompt
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    # build a list of possible router destinations
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    # Define a default destination chain
    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    # Format the router prompt
    router_template = router_prompt_template.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    # Define the router chains using the router prompt
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    # Create the chain using the router, we need to give it a list of possible destinations (chains)
    # as well as the default chain (for the situations when the LLM cannot determine how to route
    # the user request
    chain = MultiPromptChain(router_chain=router_chain,
                             destination_chains=destination_chains,
                             default_chain=default_chain,
                             verbose=True
                             )
    return chain


def run_app():
    """
    Call the different types of chains and run them
    """

    llm = ChatOpenAI(temperature=0.9, model=llm_model)
    product = "Queen Size Sheet Set"
    # LLM Chain
    basic_llm_chain(product, llm)

    # Simple Sequential Chain
    simple_sequential(product, llm)

    # Sequential Chain
    filepath="data/ProductsReview.csv"
    df_product_reviews = load_data(filepath)
    review_index = 4
    sequential_chain(df_product_reviews, review_index, llm)

    # Routing Chain
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    templates = router_chain_templates()
    prompt_infos = router_chain_prompt_infos(templates)
    router_prompt_template = build_router_template_prompt()
    chain = build_router_chain(prompt_infos, router_prompt_template, llm)
    print(chain.invoke({"input":"What is black body radiation?"}))
    print(chain.invoke({"input":"What is 2 + 2"}))
    print(chain.invoke({"input":"Who is R2D2"}))


run_app()

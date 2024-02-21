# Examples of how to use Langchain memory to interact with stateless LLMs
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


def define_conversation(verbose, memory, llm):

    conversation = ConversationChain(
        llm=llm,
        memory=memory,  # Add the memory to the conversation chain
        verbose=verbose  # Enables getting additional details on the conversation
    )
    return conversation


def conversation_buffer_memory(conversation, memory):
    # Show how a simple conversation buffer memory can be used to store history of a conversation

    print('---------- Buffer ------------')

    conversation.predict(input="Hi, my name is Andrew")
    conversation.predict(input="What is 1+1?")
    conversation.predict(input="What is my name?")

    # Buffer contains the conversation with the LLM as memory was added to the conversation chain
    print(f'first memory buffer: {memory.buffer}')
    print(f'first memory vars:{memory.load_memory_variables({})}')

    # Reset memory
    memory = ConversationBufferMemory()

    # Modify to context by adding new inputs and outputs explicitly
    memory.save_context({"input": "Hi"},
                        {"output": "What's up"})

    print(f'second memory buffer:{memory.buffer}')
    print(f'second memory vars:{memory.load_memory_variables({})}')

    # continue to add stuff to memory
    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})
    print(f'third memory vars: {memory.load_memory_variables({})}')


from langchain.memory import ConversationBufferWindowMemory
def conversation_buffer_window_memory(conversation):
    print('---------- Window buffer ------------')
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "Hi"},
                        {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})
    #  Only remembers last interaction since K = 1
    print(f'fourth memory vars: {memory.load_memory_variables({})}')
    # Same for LLM interactions
    conversation.predict(input="Hi, my name is Andrew")
    conversation.predict(input="What is 1+1?")
    conversation.predict(input="What is my name?")
    print(f'fifth memory vars: {memory.load_memory_variables({})}')


from langchain.memory import ConversationTokenBufferMemory
def conversation_token_buffer_memory(memory):
    print('---------- Token buffer ------------')
    #  Token limit has been set in the memory so it will not longer remember (or print) the whole thing
    memory.save_context({"input": "AI is what?!"},
                        {"output": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"},
                        {"output": "Beautiful!"})
    memory.save_context({"input": "Chatbots are what?"},
                        {"output": "Charming!"})
    print(f'sixth memory vars: {memory.load_memory_variables({})}')


from langchain.memory import ConversationSummaryBufferMemory

def conversation_summary_buffer_memory(conversation, memory):
    print('---------- Summary buffer ------------')

    # create a long string
    schedule = "There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo."


    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})
    memory.save_context({"input": "What is on the schedule today?"},
                        {"output": f"{schedule}"})

    print(f'seventh memory vars: {memory.load_memory_variables({})}')

    conversation.predict(input="What would be a good demo to show?")
    print(f'eight memory vars: {memory.load_memory_variables({})}')
    # In the print above, we see that the LLM incorporates the AI response to the memory (literal up until the
    # max number of tokens we have specified and anything before the latest response is summarized

def run_app():

    llm_model = "gpt-3.5-turbo"

    llm = ChatOpenAI(temperature=0.0, model=llm_model)

    memory = ConversationBufferMemory()  # stores the conversation
    conversation = define_conversation(True, memory, llm)
    conversation_buffer_memory(conversation, memory)

    memory = ConversationBufferWindowMemory(k=1)  # store only last interaction
    conversation = define_conversation(False, memory, llm)
    conversation_buffer_window_memory(conversation)

    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)  # store only last 50 tokens, Need to specify LLM since different LLMs have different token counters
    # conversation = define_conversation(False, memory, llm)
    conversation_token_buffer_memory(memory)

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=400)  # Used to generate a summary of Conv., but store only last 100 token
    conversation = define_conversation(True, memory, llm)
    conversation_summary_buffer_memory(conversation, memory)


run_app()
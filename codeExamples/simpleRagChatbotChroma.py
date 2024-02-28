from langchain_openai import ChatOpenAI
# Imports to load data into the vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# End of imports to load data into the vector DB
# Imports to build the Chatbot chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
# End of imports to build the Chatbot chain


llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0, model=llm_model)


def load_pdf(path):
    """
    Load a PDF file into a list of Document objects. Each pape in the PDF is a Document object.
    :param path: Path to the PDF file
    :return: List of Document objects
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages


def split_pdfs():
    """
    Split 3 documents into chunks
    :return: splits which is the chunks of the combined documents
    """""

    docs = []
    docs.extend(load_pdf("data/MachineLearning-Lecture01.pdf"))
    docs.extend(load_pdf("data/MachineLearning-Lecture02.pdf"))
    docs.extend(load_pdf("data/MachineLearning-Lecture03.pdf"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    splits = text_splitter.split_documents(docs)

    print(f'len of splits: {len(splits)}')

    return splits


def load_to_chromaDB(splits, persist_directory):
    """
    Load the document splits (chunks) into the Chroma DB
    :param splits: document chunks
    :param persist_directory: Directory to save the Chroma DB
    :return: Vectorstore object
    """
    vectorstore = Chroma.from_documents(documents=splits,
                                        persist_directory=persist_directory,
                                        embedding=OpenAIEmbeddings())
    print(f'Number of items in Chroma collection: {vectorstore._collection.count()}')
    return vectorstore

def define_memory(memory_key):
    """
    Define the memory to be used by the Chatbot chain. The memory is used by the Chatbot chain to store the chat history
     and return the last message in the chat
    :param memory_key: Key to be used as identifier for the memory
    :return: ConversationBufferMemory object
    """
    return ConversationBufferMemory(memory_key=memory_key, return_messages=True)

def get_answer_from_llm(question, vectorstore, memory, chain_type="stuff",
                        return_source_documents=False, return_generated_question=False):
    """
    Get the answer to a question from the Chatbot chain. Note that we are not providing a condense_question_prompt
    and thus the langchain will use its default prompt for ConversationalRetrievalChain
    :param question: Question to be answered.
    :param vectorstore: Vectorstore object
    :param memory: ConversationBufferMemory object
    :param chain_type: Type of chain to be used. Set to stuff by default
    :param return_source_documents: Whether to return the source documents. Defaults to false
    :param return_generated_question: Whether to return the generated question. Defaults to false
    """

    qa_chain = ConversationalRetrievalChain.from_llm(llm,
                                                     retriever=vectorstore.as_retriever(),
                                                     chain_type=chain_type,
                                                     memory=memory,
                                                     return_source_documents=return_source_documents,
                                                     return_generated_question=return_generated_question
                                                     )

    result = qa_chain.invoke({"question": question})
    if result:
        print(f'Question:\n {question}')
        print(f'Answer:\n {result["answer"]}')
        print(f'Chat History:\n {result["chat_history"]}')
    else:
        print("No result returned")


def app_run():
    """
    Load 3 PDFs and split them into chunks that are then loaded into a vector store. Define a memory to be used 
    by the chain so we cna ask follow up questions. Define and run chain with a couple of questions
    :return: 
    """""

    splits = split_pdfs()
    save_data_to_disk = False  # change to true to persist chroma data to disk
    persist_directory = None
    if save_data_to_disk:
        persist_directory = 'chromaDB/docs/'
    vectorstore = load_to_chromaDB(splits, persist_directory)

    memory = define_memory("chat_history")

    question = "Is probability a class topic?"
    get_answer_from_llm(question, vectorstore, memory)
    question = "why are those prerequisites needed?"
    get_answer_from_llm(question, vectorstore, memory)

app_run()
from langchain_openai import ChatOpenAI
# Imports to load data into the vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# End of imports to load data into the vector DB

# Memory for chatbot
from langchain.memory import ConversationBufferMemory
# Chatbot chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0, model=llm_model)


def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()  # Each pape in the PDF is a Document object
    return pages


def split_pdfs():
    docs = []
    docs.extend(load_pdf("data/MachineLearning-Lecture01.pdf"))
    docs.extend(load_pdf("data/MachineLearning-Lecture02.pdf"))
    docs.extend(load_pdf("data/MachineLearning-Lecture03.pdf"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    splits = text_splitter.split_documents(docs)

    print(f'len of splits: {len(splits)}')

    return splits


def load_to_chromaDB(splits, persist_directory):
    vectorstore = Chroma.from_documents(documents=splits,
                                        persist_directory=persist_directory,
                                        embedding=OpenAIEmbeddings())
    print(f'Number of items in Chroma collection: {vectorstore._collection.count()}')
    return vectorstore

def define_memory(memory_key):
    return ConversationBufferMemory(memory_key=memory_key, return_messages=True)

def get_answer_from_llm(question, vectorstore, memory, chain_type="stuff",
                        return_source_documents=False, return_generated_question=False):

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


def define_prompt():
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    return PromptTemplate.from_template(input_variables=["context", "question"],template=template)

def app_run():

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
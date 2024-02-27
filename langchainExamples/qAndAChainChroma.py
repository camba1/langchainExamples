from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Imports to load data into the vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# End of imports to load data into the vector DB

# Traditional QA chain import
from langchain.chains import RetrievalQA
# End of traditional QA chain import


# Import specific to the LCEL style of QA chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
# End of imports specific to the LCEL style of QA chains

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0, model=llm_model)


def load_pdf(path):
    """
    Load a PDF file into a list of Document objects. Each pape in the PDF is a Document object.
    :param path: Path to the PDF file
    :return: List of Document objects
    """
    loader = PyPDFLoader(path)
    pages = loader.load()  # Each pape in the PDF is a Document object
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


def get_answer_from_llm(question, vectorstore,
                        return_source_documents=False, chain_type_kwargs=None, chain_type="stuff"):
    """
    Get the answer to a question from the Chatbot chain.
    and thus the langchain will use its default prompt for ConversationalRetrievalChain
    :param chain_type_kwargs: Additional arguments to be passed to the chain like the prompt template to be used
    :param question: Question to be answered.
    :param vectorstore: Vectorstore object
    :param chain_type: Type of chain to be used. Set to stuff by default
    :param return_source_documents: Whether to return the source documents. Defaults to false

    """
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=return_source_documents,
        chain_type_kwargs=chain_type_kwargs,
        chain_type=chain_type
    )

    result = qa_chain.invoke({"query": question})
    print(f"\n---------chain type: {chain_type}---------")
    print(f'Query:\n {result["query"]}')
    print(f'Result:\n {result["result"]}')
    if return_source_documents:
        # print first source document, there can be many
        print(f'++++++Source documents:\n {result["source_documents"][0]}')
    return result


def define_prompt_template():
    """
    Define the template of the prompt to be passed to the model
    :return: Prompt template object
    """
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer.
     
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    return PromptTemplate.from_template(template)


def lcel_get_answer_from_llm(question, vectorstore):
    """
    Pulls a prompt template from the langchain hub and uses it to create a retrieval chain using the
    create_retrieval_chain LCEL chain constructor.The retrieval chain is then used to retrieve
    the answer to the question.
    :param question: Question to be answered
    :param vectorstore: Vector store where we have the necessary context
    :return: Answer for the question
    """

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = vectorstore.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})
    print(f"\n---------NEW Q&A Retrieval---------")
    print(f'Query:\n {question}')
    print(f'Result:\n {result["answer"]}')
    print(f'First context:\n {result["context"][0]}')
    return result


def app_run():
    """
    Load 3 PDFs and split them into chunks that are then loaded into a vector store.
    Define a prompt template and use it to create a chain with a custom prompt.
    Define and run a traditional Q and A chain with a couple of questions.
    Note: RetrievalQA does not automatically have memory by default (fire and forget)
    Finally, define and run a new chain created using a LCEL chain constructor
    """

    question = "What are major topics for this class?"
    splits = split_pdfs()
    save_data_to_disk = False  # change to true to persist chroma data to disk
    persist_directory = None
    if save_data_to_disk:
        persist_directory = 'chromaDB/docs/'
    vectorstore = load_to_chromaDB(splits, persist_directory)

    #  Basic Call
    get_answer_from_llm(question, vectorstore)
    # Custom Prompt & return the source document of the information
    prompt_template = define_prompt_template()
    get_answer_from_llm(question, vectorstore, chain_type_kwargs={"prompt": prompt_template}, return_source_documents=True)

    # Different chain types
    get_answer_from_llm(question, vectorstore, chain_type="map_reduce")
    get_answer_from_llm(question, vectorstore, chain_type="refine")

    # Using an LCEL chain constructor
    lcel_get_answer_from_llm(question, vectorstore)


app_run()

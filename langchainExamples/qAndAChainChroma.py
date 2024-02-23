from langchain_openai import ChatOpenAI
# Imports to load data into the vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# End of imports to load data into the vector DB
# Traditional QA chain import
from langchain.chains import RetrievalQA
# End of traditional QA chain import

from langchain.prompts import PromptTemplate

# Import specific to the LCEL style of QA chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
# End of imports specific to the LCEL style of QA chains

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

def get_answer_from_llm(question, vectorstore,
                        return_source_documents=False, chain_type_kwargs=None,chain_type="stuff"):
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

def define_prompt():

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
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever=vectorstore.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})
    # result = qa_chain.run(question)
    print(f"\n---------NEW Q&A Retrieval---------")
    print(f'Query:\n {question}')
    print(f'Result:\n {result["answer"]}')
    print(f'First context:\n {result["context"][0]}')
    return result

def app_run():

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
    prompt = define_prompt()
    get_answer_from_llm(question, vectorstore, chain_type_kwargs={"prompt": prompt}, return_source_documents=True)

    # Different chain types
    get_answer_from_llm(question, vectorstore, chain_type="map_reduce")
    get_answer_from_llm(question, vectorstore, chain_type="refine")

    # By default, RetrievalQA does not automatically have memory, it has to be added to the prompt manually

    # LCEL style of QA chains
    lcel_get_answer_from_llm(question, vectorstore)

app_run()
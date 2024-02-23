from langchain_openai import OpenAIEmbeddings
import numpy as np

def generate_emebdings(my_text):
    embeddings = OpenAIEmbeddings()
    embedding = embeddings.embed_query(my_text)
    # print(f'Embeddings: {embedding}')
    return embedding

def compare_simple_embeddings():
    embedding1 = generate_emebdings("i like dogs")
    embedding2 = generate_emebdings("i like canines")
    embedding3 = generate_emebdings("the weather is ugly outside")
    # compare the distance of two embeddings  using dot product
    print(f'np.dot 1 v 2:  {np.dot(embedding1, embedding2)}')
    print(f'np.dot 1 v 3:  {np.dot(embedding1, embedding3)}')
    print(f'np.dot 2 v 3:  {np.dot(embedding2, embedding3)}')



from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()  # Each pape in the PDF is a Document object
    # print(f'pages in PDF: {len(pages)}')
    # print(f'Page metadata: {pages[0].metadata}')
    return pages


def split_pdfs():
    docs = []
    docs.extend(load_pdf("data/MachineLearning-Lecture01.pdf"))
    docs.extend(load_pdf("data/MachineLearning-Lecture01.pdf")) # loaded duplicate on purpose
    docs.extend(load_pdf("data/MachineLearning-Lecture02.pdf"))
    docs.extend(load_pdf("data/MachineLearning-Lecture03.pdf"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    splits = text_splitter.split_documents(docs)

    print(f'len of splits: {len(splits)}')

    return splits


from langchain_community.vectorstores import Chroma

def load_to_chromaDB(splits, persist_directory):
    # Chroma will do the embeddings using the function provided, no need to do
    # the embeddings separately
    # set persist_directory = None to avoid writing data to disk
    vectorstore = Chroma.from_documents(documents=splits,
                                        persist_directory=persist_directory,
                                        embedding=OpenAIEmbeddings())
    print(f'Number of items in Chroma collection: {vectorstore._collection.count()}')
    return vectorstore

def search_db(query, reload_chroma_data, persist_directory, vectorstore = None, max_results = 3, filter = None):
    if reload_chroma_data or vectorstore is None:
        vectorstore = Chroma.load_local(persist_directory, OpenAIEmbeddings())
    results = vectorstore.similarity_search(query, k=max_results, filter=filter)
    print("\n---------SEARCH RESULTS---------")
    print(f'Query: {query}')
    print(f'Results from query: {len(results)}')
    print(f'++++++First result:\n {results[0].page_content[:100]}')
    print(f'++++++Second result:\n {results[1].page_content[:100]}')
    print(f'++++++First metadata:\n {results[0].metadata}')
    print(f'++++++Second metadata:\n {results[1].metadata}')
    return results

def mmr_search(vectorstore, query):
    results = vectorstore.max_marginal_relevance_search(query)
    print("\n---------MMR RESULTS---------")
    print(f'Query: {query}')
    print(f'Results from query: {len(results)}')
    print(f'++++++First result:\n {results[0].page_content[:100]}')
    print(f'++++++Second result:\n {results[1].page_content[:100]}')
    return  results


from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
def self_query(query, vectorstore):
#      Specify the metadata we want to use for the self query
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `data/MachineLearning-Lecture01.pdf`, `data/MachineLearning-Lecture02.pdf`, or `data/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]

    llm_model = "gpt-3.5-turbo-instruct"
    llm = OpenAI(temperature=0, model=llm_model)
    document_content_description = "Lecture notes"
    retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description,
                                            metadata_field_info, verbose=True)

    results = retriever.get_relevant_documents(query)

    print("\n---------SELF-QUERY RESULTS---------")
    print(f'Query: {query}')
    for d in results:
        print(f'++++++Result:\n {d.page_content[:100]}')
        print(d.metadata)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
def compression(query, vectorstore):
    llm_model = "gpt-3.5-turbo-instruct"
    llm = OpenAI(temperature=0, model=llm_model)
    compressor = LLMChainExtractor.from_llm(llm)
    # Create hte retriever. Note that beyond compression, we are also asking it to use MMR as the search type
    # this is optional.
    retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                base_retriever=vectorstore.as_retriever(search_type="mmr"))
    results = retriever.get_relevant_documents(query)
    for d in results:
        print(f'++++++Result:\n {d.page_content[:100]}')
        print(d.metadata)

def app_run():
    # compare_simple_embeddings()

    splits = split_pdfs()

    save_data_to_disk = False  # change to true to persist chroma data to disk
    persist_directory = None
    if save_data_to_disk:
        persist_directory = 'chromaDB/docs/'
    vectorstore = load_to_chromaDB(splits, persist_directory)

    # Note how some of these searches return the same thing twice as we loaded one of the pdfs twice
    search_db("is there an email where i can ask for help",  False,
              persist_directory, vectorstore)

    search_db("what did they say about matlab?", False,
              persist_directory, vectorstore, 5)

    # This returns data from multiple lectures even-though we asked for just data for the third lecture
    docs = search_db("what did they say about regression in the third lecture?", False,
              persist_directory, vectorstore, 5)


    # Reduce chances of reapeted documents using relevancy search
    mmr_search(vectorstore, "what did they say about matlab?")

    # Reduce chances of unwanted documents using filters on metadata
    filter = {"source":"data/MachineLearning-Lecture03.pdf"}
    docs = search_db("what did they say about regression in the third lecture?", False,
                     persist_directory, vectorstore, 5, filter)

    # Use a self-query retriever to answer the question
    # Note how we do not have to filter on the metadata to tell the model that we want results from
    # only the third lecture. The LLM figures that out for us from the query text
    self_query("what did they say about regression in the third lecture?", vectorstore)

    # Use a compression retriever to answer the question
    # Note that as of this writing there is a warning about predict and parse being deprecated
    # that can be ignored when using compression
    compression("what did they say about matlab?", vectorstore)

app_run()
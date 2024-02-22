from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

def simple_example():
    chunk_size =26
    chunk_overlap = 4

    # You can specify multiple separators here using the separator keyword.
    # under the covers this is using a regex. The recursive  splitter is the recommended for
    # many scenarios
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # character spliter splits on new lines by default
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print("--------Simple Example------")
    text1 = 'abcdefghijklmnopqrstuvwxyz'
    print(f'R split text1: {r_splitter.split_text(text1)}')
    text2 = 'abcdefghijklmnopqrstuvwxyz123456'
    print(f'R split text2: {r_splitter.split_text(text2)}')
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    print(f'R split text3: {r_splitter.split_text(text3)}')
    # There are no new lines in text3 string so the character splitter does nothing
    print(f'C split text3: {c_splitter.split_text(text3)}')

    # specify the separator so that it will actually split the text that has no new lines
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=' '
    )
    print(f'C split text3 with new separator: {c_splitter.split_text(text3)}')

from langchain_community.document_loaders import PyPDFLoader

def split_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)

    print("--------PDF Splitter------")
    print(f'len of original doc {len(pages)}')
    print(f'len of split Docs {len(docs)}')
    print(f'fifth split doc metadata (note that it came from the 1st page in the original doc): {docs[5].metadata}')


from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader

def context_aware_splitting(path):

    # Cannot use the markdown loader as that removes the markdown charters (e.g. #, ##, etc)
    # so we use the basic text loader instead
    # loader = UnstructuredMarkdownLoader(path)
    loader = TextLoader(path)
    data = loader.load()
    markdown_document = data[0].page_content

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    print("--------Markdown Splitter------")
    print(len(md_header_splits))
    print(f'First split: {md_header_splits[0]}')
    print(f'Second split {md_header_splits[1]}')

def app_run():
    simple_example()
    split_pdf("data/MachineLearning-Lecture01.pdf")
    context_aware_splitting("data/mardownDocToSplit.md")

app_run()
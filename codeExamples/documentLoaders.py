# Import for pdf Loader
from langchain_community.document_loaders import PyPDFLoader

# Imports for youTube loader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# Import for web loader
from langchain_community.document_loaders import WebBaseLoader

def load_pdf(path):
    """
    Load a PDF file`
    :param path: Path to the PDF to be loaded
    :return: List of document pages
    """
    loader = PyPDFLoader(path)
    pages = loader.load()   # Each pape in the PDF is a Document object
    print(f'pages in PDF: {len(pages)}')
    print(f'first page (partial): \n {pages[0].page_content[0:500]}')
    print(f'Page metadata: {pages[0].metadata}')
    return pages


def youTubeLoader(url, save_dir):
    """
    Load a YouTube video audio from its URL. Pass the video to OpenAi's whisper parser to transcribe
     the video and return the text
    :param url: Url to the YouTube video
    :param save_dir: Folder to which the video's audio will be saved
    """
    #
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    print(docs[0].page_content[0:500])



def url_loader(url):
    """
    This load a web page context. However, given that it is basically scrapping it,
    there is a lot of white space that could be removed in post-processing
    :param url: Url to load
    :return:
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f'length of document: {len(docs[0].page_content)}')
    print(f'URL loader first page: \n {docs[0].page_content[:500]}')

def app_run():
    """
    Loads a PDF, a YouTube video and a web page
    """
    load_pdf("data/MachineLearning-Lecture01.pdf")

    youtube_url_to_cs_course = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir = "output_docs/youtube/"
    youTubeLoader(youtube_url_to_cs_course, save_dir)

    web_url = "https://github.com/camba1/simplifyyourlife/blob/master/README.md"
    url_loader(web_url)

app_run()
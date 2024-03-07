# Simple LLM Chat Examples

This repo is a compilation of simple examples to run LLM based applications.
Note that for the majority of these examples to work, you will to have the openAI API key set in your environment.


## Samples

- **Chainlit**:
  - **chainlitChatbot**: Simple chatbot application using Chainlit (WebUI), langChain and OpenAI.
Run with `chainlit run chainlitChatBot.py` . Note that chainlit will create a cuople of files on the first run
including the config.toml where you can disable sending telemetry back to Chainlit and set additional configuration.
It also creates a chainlit.md which is used as the chatbot Readme page.
- **langChainExamples**: Folder that contains a number of simple Langchain examples. These are all run from the command line
  - **agent**: Create an Agent to run ll-math tool and solve a math question
  - **agentLCEL**: Create an Agent to run ll-math tool and solve a math question. This agent is build using LCEL
  - **agentPlanExecute**: Create s Plan and Execute type of Agent to run a math solving tool
  - **chunkSplitters**: Examples on hos to split markdown and pdf files
  - **conversationMemory**: Examples on how to use the different conversation memory types
  - **documentLoaders**: Examples on how to load PDFs, YouTube videos (and generating the transcript) and Web content
  - **embeddingsAndRetrieval**: How to create embeddings, load them into a vector store (chroma) and retrieve the data from
vector store
  - imageQandAChain: Pass an image or an image URL to OpenAi Vision API and and questions about the image
  - **multistepChains**: How to setup multi-step chains
  - **outputParser**: Creating an output parser to format an LLM response
  - **panelRagChatBot**: Full (mostly) RAG chatbot implementation, including Web UI using Panel. Run using `panel serve --show panelRagChatBot.py `
  - **promptBuilding**: Example of building different prompts for a basic chain
  - **qAndAChainChroma**: Simple Q&A chatbot implementation
  - **simpleRagChatbotChroma**: A basic chat that uses RAG along with a vector store (Chroma)
- **gradioChatbot**: Simple chat application that was built using Gradio (WebUI), langchain and OpenAI
Run with ` python gradioChatbot.py` or to have watching enabled run `gradio gradioChatbot.py`
- **openAiFunctions**: Examples of calling OpenAI functions using  OpenAI directly
- **simpleGradio**: A simple Gradio frontend. Not connected to anything. Run with ` python simpleGradio.py`
- **simpleOpenAi**: Simple program to interact with openAI using the OpenAi library directly
- **streamlitChatBot**: Simple chat application that was built using Streamlit (WebUI), langchain and OpenAI.
Run with `streamlit run streamlitChatBot.py`. This implements simple "streaming" to simulate the chatbot response being typed

  
### Getting started
To run the different examples, clone the repo, create your virtual environment and install the requirements

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  
```

Additionally, you will need an OpenAi API Key for the majority of the examples. Once you obtain one, set it up in your
shell as an environment variable

```shell
export OPENAI_API_KEY=<your API KEY>
```

## Langsmith

If you want to send traces of your chain runs to LangSmith, you can set the following environment variables:

```shell
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY=<YOUR LANGSMITH_API_KEY >
  export LANGCHAIN_PROJECT="Test Project"
```
Note that LANGCHAIN_PROJECT is optional. If a project is not specified, the trace goes into the "default" project

## Example specific installation notes

For the YouTube downloader code (langchainExamples/simpleDocumentLoaders.py), ensure that you have ffmpeg installed
and in the system path. The easiest way to do this is using brew (on Mac):

```shell
brew install ffmpeg
```


For the projects that use ChromaDB (most RAG examples), you will need to install it in order to run them.
If you run into the error:

```shell
error: command '/usr/bin/clang' failed with exit code 1
ERROR: Could not build wheels for chroma-hnswlib, which is required to install pyproject.toml-based projects
```
Then set the following environment variable and try to install again:

```shell
export HNSWLIB_NO_NATIVE=1 
```


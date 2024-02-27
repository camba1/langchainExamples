# Simple LLM Chat Examples

This repo is a compilation of simple examples to run LLM based applications.
Note that for the majority of these examples to work, you will to have the openAI API key set in your environment.

```shell
export OPENAI_API_KEY=<your API KEY>
```

## Samples

- streamlitChatBot: Simple chat application that was built using Streamlit (WebUI), langchain and OpenAI. Run with `streamlit run streamlitChatBot.py`
- simpleOpenAi: Simple program to interact with openAI using the OpenAi library directly
- simpleGradio: A simple Gradio frontend. Not connected to anything
- gradioChatbot: Simple chat application that was built using Gradio (WebUI), langchain and OpenAI
- chainlitChatbot: Simple chatbot application using Chainlit (WebUI), langChain and OpenAI
Run with ` python gradioChatbot.py` or to have watching enabled run `gradio gradioChatbot.py`
This implements simple "streaming" to simulate the chatbot response being typed
- langChainExamples: Folder that contains a number of simple Langchain examples. These are all run from the command line
  - agent: Create an Agent to run ll-math tool and solve a math question
  - agentLCEL: Create an Agent to run ll-math tool and solve a math question. This agent is build using LCEL
  - agentPlanExecute: Create s Plan and Execute type of Agent to run a math solving tool
  - chunkSplitters: Examples on hos to split markdown and pdf files
  - conversationMemory: Examples on how to use the different conversation memory types
  - documentLoaders: Examples on how to load PDFs, YouTube videos (and generating the transcript) and Web content
  - embeddingsAndRetrieval: How to create embeddings, load them into a vector store (chroma) and retrieve the data from
vector store
  - multistepChains: How to setup multi-step chains
  - outputParser: Creating an output parser to format an LLM response
  - panelRagChatBot: Full (mostly) RAG chatbot implementation, including Web UI using Panel. Run using `panel serve --show panelRagChatBot.py `
  - qAndAChainChroma: Simple Q&A chatbot implementation
  - promptBuilding: Example of building different prompts for a basic chain
  - simpleRagChatbotChroma: A basic chat that uses RAG along with a vector store (Chroma)
  

Install the requirements 
```shell
pip install -r requirements.txt  
```

## Langsmith

If you want to send traces of your chains to LangSmith, you can set the following environment variables:

```shell
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY=<YOUR LANGSMITH_API_KEY >
  export LANGCHAIN_PROJECT="Test Project"
```
Note that LANGCHAIN_PROJECT is optional. If a project is not specified, the trace goes into the "default" project

For the YouTube downloader code (langchainExamples/simpleDocumentLoaders.py), ensure that you have ffmpeg installed
and in the system path. The easiest way to do this is using brew (on Mac):
```shell
brew install ffmpeg
```


When installing chormaDB, if you run into the error:
```shell
error: command '/usr/bin/clang' failed with exit code 1
ERROR: Could not build wheels for chroma-hnswlib, which is required to install pyproject.toml-based projects
```
The set the following environment variable and try again:

```shell
export HNSWLIB_NO_NATIVE=1 
```
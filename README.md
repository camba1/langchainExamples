# Simple LLM Chat Application

This is a simple chat application that was built using Streamlit, langchain and OpenAI (as the LLM)

Install the requirements 
```shell
pip install -r requirements.txt  
```

Run using: 
```shell
streamlit run streamlitChatBot.py
```


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
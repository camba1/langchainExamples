import panel as pn
import param

# --------- Chatbot Logic ------------
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import   ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

class RagChatBot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, default_file, chain_type, llm, **params):
        super(RagChatBot, self).__init__(**params)
        self.panels = []
        self.loaded_file = default_file
        self.llm = llm
        self.chain_type = chain_type
        self.qa = self.load_db(self.llm, self.loaded_file, self.chain_type, 4)


    def call_load_db(self, count):
        """
        Load a new file and update the chatbot.
        :param count: Help determine if we load the default file or a file provided by the user
        :return: updated filename to be displayed in the UI
        """
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("output_docs/temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = self.load_db(self.llm,"output_docs/temp.pdf", self.chain_type, 4)
            button_load.button_style = "solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        """
        If the user provides a query, this function is called to handle the query and return the answer.
        It also updates the chat history and the panels to display the query and the answer.
        Finally, it updates the db_query and db_response parameters to display the last query and the response.
        :param query: Question posted by the user
        :return:
        """
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa.invoke({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, styles={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  # clears loading indicator when cleared
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        """
        This method populates the last vector database query or a default message if there has been no DB access yet
        :return: Last query sent to the DB
        """
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query)
        )

    @param.depends('db_response', )
    def get_sources(self):
        """
        This method populates the last vector database response or a empty if there has been no DB access yet
        :return: Last response from the DB or empty
        """
        if not self.db_response:
            return
        rlist = [pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        """
        This method populates the chat history or a empty if there has been no DB access yet
        :return: Chat history
        """
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist = [pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self, count=0):
        """
        Clears the variable that holds the chat history
        :param count:
        :return:
        """
        self.chat_history = []
        return

    @staticmethod
    def load_db(llm, file, chain_type, k):
        """
        Loads a file contents to the vector DB and prepares a conversation chain to interact with the LLM
        :param llm: LLM to use for the conversation chain
        :param file: File to load to the vector DB
        :param chain_type: Type of conversation chain to use.
        :param k: Number of results to return from the vector DB
        :return: Conversational retrieval chain object
        """
        # load documents
        loader = PyPDFLoader(file)
        documents = loader.load()
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        # create vector database from data
        vectorstore = Chroma.from_documents(documents=docs,
                                            embedding=OpenAIEmbeddings())
        # define retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # create a chatbot chain. Memory is managed externally.
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa


# --------- Main UI Code ------------

llm_model = "gpt-3.5-turbo"
default_file = "data/MachineLearning-Lecture01.pdf"
llm = ChatOpenAI(temperature=0, model=llm_model)
chain_type = "stuff"

rag_chat_bot = RagChatBot( default_file, chain_type, llm)

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(rag_chat_bot.clr_history)
inp = pn.widgets.TextInput( placeholder='Enter text here…')

bound_button_load = pn.bind(rag_chat_bot.call_load_db, button_load.param.clicks)
conversation = pn.bind(rag_chat_bot.convchain, inp)

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2= pn.Column(
    pn.panel(rag_chat_bot.get_lquest),
    pn.layout.Divider(),
    pn.panel(rag_chat_bot.get_sources),
)
tab3= pn.Column(
    pn.panel(rag_chat_bot.get_chats),
    pn.layout.Divider(),
)
tab4=pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
    pn.layout.Divider(),

)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
)
dashboard.servable()
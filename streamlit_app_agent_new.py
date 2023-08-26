from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
import tempfile
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
import pandas as pd
import os
import csv
import gspread
from google.oauth2 import service_account
import base64
from datetime import datetime
from pytz import timezone
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("socialai.jpg")
file = r'dealer_1_inventry.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", k=8)
# Streamlit UI setup
st.info(" We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing a environment to check offerings and also check Our website [engane.ai](https://funnelai.com/). This test application answers about Inventry, Business details, Financing and Discounts and Offers related questions. [here](https://github.com/buravelliprasad/streamlit/blob/main/dealer_1_inventry.csv) is a inventry dataset explore and play with the data.")
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
def save_chat_to_google_sheets(user_name, user_input, output, timestamp):
    try:
        # Connect to Google Sheets using service account credentials
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        gc = gspread.authorize(credentials)
        
        # Get the Google Sheet by URL
        sheet_url = st.secrets["public_gsheets_url"]
        sheet = gc.open_by_url(sheet_url)
        
        # Select the desired worksheet
        worksheet = sheet.get_worksheet(0)  # Replace 0 with the index of your desired worksheet
    
        data = [timestamp, user_name, user_input, output]
        worksheet.append_row(data)
        # st.success("Data saved to Google Sheets!")
    except Exception as e:
        st.error(f"Error saving data to Google Sheets: {str(e)}")
        
# Model details
tool = create_retriever_tool(
    retriever, 
    "search_car_dealership_inventory",
    "Searches and returns documents regarding the car inventory."
)
tools = [tool]
llm = ChatOpenAI(temperature = 0)
memory_key = "history"
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
system_message = SystemMessage(
        content=(
            "You are a business development manager role \
working in a car dealership you get a text enquiry regarding inventory,\
business details and finance. Given the following conversation and a follow up question,\
'You should answer in a style that is American English in a calm and respectful tone.' \
If you do not know the answer, reply with 'I am sorry'.\
Do your best to answer the questions.\
Feel free to use any tools available to look up\
relevant information, only if neccessary"
        )
)
prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
from langchain.agents import AgentExecutor
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

print("this code block running every time")


if 'agent_executor' not in st.session_state:
	agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
	st.session_state.agent_executor = agent_executor
else:
	agent_executor = st.session_state.agent_executor


response_container = st.container()
container = st.container()
chat_history=[]
def conversational_chat(user_input):
# question=("my name is prasad")
# agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
    result = agent_executor({"input":user_input})
    # chat_history= st.session_state.chat_history
    # answer=result["output"]
    st.session_state.chat_history.append((user_input, result["output"]))
    # st.session_state.chat_history.append((user_input, answer))
    return result["output"]
#     result=qa({"question":question, "chat_history":chat_history})
#     chat_history = [(question, result["answer"])]
#     result=qa({"question":question, "chat_history":chat_history})
    # print(result["output"])
    # chat_history.append((question, result["output"])) 

# chat_history=[] 
# def conversational_chat(query):
#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result["answer"])) 
#     st.session_state.history.append((query, result["answer"]))
#     return result["answer"]
# chat_history=[] 
# def conversational_chat(user_input):
#     result = qa({"question": user_input, "chat_history": st.session_state.chat_history})
#     # print("this is testing chat history",result)
#     st.session_state.chat_history.append((user_input, result["answer"]))
#     return result["answer"]
    
with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input:
        output = conversational_chat(user_input)
       
        # Get the current UTC timestamp
        utc_now = datetime.now(timezone('UTC'))
        
        # Display conversation history with proper differentiation
        with response_container:
            for i, (query, answer) in enumerate(st.session_state.chat_history):
                message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(answer, key=f"{i}_answer", avatar_style="thumbs")
        
        Save conversation to Google Sheets along with user name and UTC timestamp
        if st.session_state.user_name:
            try:
                save_chat_to_google_sheets(st.session_state.user_name, user_input, output, utc_now.strftime('%Y-%m-%d-%H-%M-%S'))
            except Exception as e:
                st.error(f"An error occurred: {e}")
            # save_chat_to_google_sheets(st.session_state.user_name, user_input, output, utc_now.strftime('%Y-%m-%d-%H-%M-%S'))

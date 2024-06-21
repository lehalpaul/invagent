import streamlit as st
import os
import openai
import sys
sys.path.append('/path/to/langchain')

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain.document_loaders import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def main():
    
    openai.api_key=os.environ["OPENAI_API_KEY"]
    st.title("Auto Dealership AI Assistant")

   
    if "messages" not in st.session_state:
        st.session_state.messages = []

   
    with st.form("user_input_form"):
        user_input = st.text_input("How can I assist you today?", key="input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = run_conversation(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

   
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
def run_conversation(question):
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-1106", streaming=True)
    loader = CSVLoader(file_path="cronicchevrolet.csv")
    data = loader.load()
    vectordb = FAISS.from_documents(data, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    template = """ You're an  AI assistant that will do analysis on given csv file {context} and answer the user {question} based on CSV Data"""
    prompt_temp = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    rag_chain = (
    {"context": retriever,"question": RunnablePassthrough()}
    | prompt_temp
    | llm
    | StrOutputParser()
)

    
    ai_response =rag_chain.invoke(question)

    return ai_response

if __name__ == "__main__":
    main()

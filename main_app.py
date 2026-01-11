import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

## api key for vector embedding 
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## load the groq api key - LLM
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
model_api_key = os.environ['GROQ_API_KEY']

## set up the streamlit 
st.title("Conversational Bot with PDF uploads and chat history")
st.write("Upload pdf and chat with their content")


if model_api_key:
    llm = ChatGroq(groq_api_key = model_api_key,model_name = "llama-3.1-8b-instant")

    session_id = st.text_input("Session ID",value = "Default_session") # take  history session id from user

    # managing history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader('Select PDF file',type = "PDF",accept_multiple_files = True)

    #processing the uploaded PDf files
    if uploaded_files:
        documents = []
        for upload_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(upload_file.getvalue()) 
                file_name = upload_file.name 

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        #split and store embeddings 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000,chunk_overlap = 500)
        splits = text_splitter.split_documents(documents)
        vectorstor = Chroma.from_documents(documents=splits,embedding = embeddings)
        retriever = vectorstor.as_retriever()

        contextualize_q_system_prompt = (
            "Given a Chat History and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood "
            "without the chat history. DO not answer the question, "
            "just reformulate it if  needed and otherwise return it as it."
        )

        contexulize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contexulize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        
        question_answer_chian = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chian)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key = "input",
            history_messages_key = "chat_history",
            output_messages_key = "answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input':user_input},
                config = {
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter the Groq API Key")
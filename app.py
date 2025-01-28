# Fix erro sqlite
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

###
from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

import chromadb


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

persist_directory = 'db'

def process_pdf(file):

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400,
    )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings()
        )
        return vector_store
    return None

def add_to_vector(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory
        )
    return vector_store

def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model = model)
    retriever = vector_store.as_retriever()
    system_prompt = '''
    Use o contexto para responder as perguntas, Se não encontrar uma resposta no contexto, explique que não há informações disponíveis.
    Responda apenas as perguntas que estejam dentro do contexto. Caso não esteja apenas responda que não pode responder.
    Responda em formato de markdown e com visualizações elaboradas e interativas. Contexto: {context}    
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm = llm,
        prompt=prompt,
    )
    
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    response = chain.invoke(
        {'input': query}
    )
    return response.get('answer')


vector_store = load_existing_vector_store()

st.set_page_config(
    page_icon='Chat PyGPT :desktop_computer:'
)
st.header('Chat com seus documentos (RAG) :desktop_computer:')

with st.sidebar:
    st.header('Upload de Arquivos')
    uploaded_files = st.file_uploader(
        label='Faça upload arquivos PDF', 
        type=['pdf'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file = uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector(
                chunks = all_chunks,
                vector_store = vector_store
                
            )



    models_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]

    select_model = st.sidebar.selectbox(
        label = 'Selecione o modelo',
        options=models_options,
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('Como posso ajudar?')

if vector_store and question:
    
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role':'user', 'content':question})

    with st.spinner('Buscando resposta...'):
        response = ask_question(
            model = select_model,
            query = question,
            vector_store = vector_store,
        )

        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})
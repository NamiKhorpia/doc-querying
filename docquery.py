#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#.\venv\Scripts\Activate.ps1
#streamlit run docquery.py
#pip install -U langchain-community

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from docx import Document

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in the .env file.")

genai.configure(api_key=api_key)

def get_file_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_name.endswith(".docx"):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_name.endswith(".txt"):
            text += uploaded_file.read().decode("utf-8")  # Handles simple text files
        else:
            st.warning(f"Unsupported file type: {file_name}")
    return text


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Use the context below to answer the question. If the context is helpful, provide the most relevant and detailed answer possible.
    If the context isn't enough, make your best educated guess and mention that the answer may not be exact.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    except ResourceExhausted:
        st.error("Quota exceeded for free tier. Please wait and try again later.")
        return None

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    if chain is None:
        return

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config("Doc Querying")
    st.header("Document-based QA system")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files",
                                          type=["pdf", "docx", "txt"],
                                          accept_multiple_files=True)

        #pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                #raw_text = get_pdf_text(pdf_docs)
                raw_text = get_file_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

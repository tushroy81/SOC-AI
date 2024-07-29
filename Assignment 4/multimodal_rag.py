import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

# Extracting Images from PDF
"""
import fitz  # PyMuPDF library
import io
from PIL import Image

def extract_images_from_pdf(pdf_path, output_folder):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Create image file
            image = Image.open(io.BytesIO(image_bytes))
            image_filename = f"{output_folder}/image_page{page_num+1}_{img_index+1}.{image_ext}"
            image.save(image_filename)
            print(f"Saved: {image_filename}")

extract_images_from_pdf("/Users/tusharsingharoy/Virtual Environments Python/Assignment4/Candlestick_Patterns_Multimodal_Data_SOC166.pdf","./Images")
"""

from langchain_experimental.open_clip import OpenCLIPEmbeddings
embedding_function = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")

import chromadb
client = chromadb.PersistentClient(path="/Users/tusharsingharoy/Virtual Environments Python/Assignment4/db/multimodal_rag")
collection = client.get_collection(
    name='multimodal_collection')
# Function to embed an image
def embed_image(image_path, embedder):
    return embedder.embed_image([image_path])

# Images Embedding
"""
image_paths = os.listdir("/Users/tusharsingharoy/Virtual Environments Python/Assignment4/images")
image_paths = [os.path.join("/Users/tusharsingharoy/Virtual Environments Python/Assignment4/images",file) for file in image_paths]
for idx, image_path in enumerate(image_paths):
    embedding = embed_image(image_path, embedding_function)
    collection.add(
        documents=[image_path],
        embeddings=embedding,
        ids=[f"image_{idx}"]
    )
"""

# Text Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = "/Users/tusharsingharoy/Virtual Environments Python/Assignment4/Candlestick_Patterns_Multimodal_Data_SOC166.pdf"
persistent_directory = os.path.join(current_dir, "db", "multimodal_rag_text2")
"""
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
# Ensure the PDF file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = rec_char_splitter.split_documents(documents)

    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
else:
    print("Vector store already exists. No need to initialize.")
"""
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding_function)
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

def image2image(img_path):
    embeddings = embedding_function.embed_query([img_path])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=1
    )
    return results

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
def image2text(image_path):
    with open(image_path, "rb") as image_file:
        # Read image file as binary
        image_binary = image_file.read()
    image_data = base64.b64encode(image_binary).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "summarise the image in text"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )
    response = llm.invoke([message]) # summary
    results = retriever.invoke([image_path])
    return results, response.content

def text2image(query):
    embeddings = embedding_function.embed_query([query])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=1
    )
    return results

def text2text(query):
    results = retriever.invoke(query)
    return results
def mod_qsn(query,chat_history):
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
    )
    message = [
        SystemMessage(content=contextualize_q_system_prompt)
    ]
    message = message + chat_history[1:]
    message.append(HumanMessage(content=query))
    result = llm.invoke(message)
    return result.content

def yes_no(query):
    prompt = (
        "From the given query, answer if the user requires an image-reply or not."
        "If user needs an image-response, answer 'yes' only or else 'no'."
        "E.g. User: 'Show me an example', Response: 'yes'"
        )
    message = [
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ]
    result = llm.invoke(message)
    return result.content

def continual_chat_rag(query,img_path,chat_history:list):
    truth = yes_no(query)
    query = mod_qsn(query,chat_history)
    if img_path is None:
        contexts = text2text(query)
        context1 = contexts[0].page_content
        context2 = contexts[1].page_content
        context3 = contexts[2].page_content
        context4 = contexts[3].page_content
        if truth[:3] == "yes":
            contexts = text2image(query)
            image_path_response = contexts["documents"][0][0]
        if truth[:2] == "no":
            image_path_response = None
    else:
        contexts = image2text(img_path)[0]
        context1 = contexts[0].page_content
        context2 = contexts[1].page_content
        context3 = contexts[2].page_content
        summary = image2text(img_path)[1]
        query = summary + f"\n{query}"
        query2 = summary + f". {query}"
        context4 = text2text(query2)[0].page_content
        if truth[:3] == "yes":
            contexts = image2image(img_path)
            image_path_response = contexts["documents"][0][0]
        if truth[:2] == "no":
            image_path_response = None
    qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise. Also avoid using 'The Text mentions/says/states' Just reply directly."
    "\n\n"
    f"Context 1: {context1}"
    "\n\n"
    f"Context 2: {context2}"
    "\n\n"
    f"Context 3: {context3}"
    "\n\n"
    f"Context 4: {context4}"
    )
    chat_history[0] = SystemMessage(content=qa_system_prompt)
    chat_history.append(HumanMessage(content=query))
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    return response, image_path_response
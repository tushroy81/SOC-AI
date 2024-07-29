import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "instigpt.pdf")
persistent_directory = os.path.join(current_dir, "db", "instigpt")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    )

# # Check if the Chroma vector store already exists
# if not os.path.exists(persistent_directory):
#     print("Persistent directory does not exist. Initializing vector store...")
# # Ensure the PDF file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

#     loader = PyPDFLoader(file_path)
#     documents = loader.load()

#     rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = rec_char_splitter.split_documents(documents)

#     # Display information about the split documents
#     # print("\n--- Document Chunks Information ---")
#     # print(f"Number of document chunks: {len(docs)}")

#     db = Chroma.from_documents(
#         docs, embeddings, persist_directory=persistent_directory)
# else:
#     print("Vector store already exists. No need to initialize.")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise. Also avoid using 'The Text mentions/says/states' Just reply directly."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to simulate a continual chat
def continual_chat(query,chat_history:list):
    # Process the user's query through the retrieval chain
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    # Update the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))
    
    return result['answer']
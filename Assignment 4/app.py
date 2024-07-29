import gradio as gr
from instigpt import continual_chat
from chat_with_docs import continual_chat_docs, clear_db, load_files, load_web
from multimodal_rag import continual_chat_rag

# INSTIGPT
chat_history2 = []
chat_history2_instigpt = []
def msg_input2(msg):
    res = continual_chat(msg,chat_history2_instigpt)
    chat_history2.append((msg,res))
    return "", chat_history2
def clear_fn2():
    global chat_history2
    chat_history2 = []
    global chat_history2_instigpt
    chat_history2_instigpt = []
    return "", chat_history2

# CHAT WITH DOCS
chat_history1 = []
chat_history1_chat_with_docs = []
def load1(files):
    if files is None:
        chat_history1.append(("No files were uploaded", "Upload files"))
    else:
        report = load_files(files)
        chat_history1.append(("Files Uploaded",report))
    return chat_history1
def msg_input1(msg):
    res = continual_chat_docs(msg,chat_history1_chat_with_docs)
    chat_history1.append((msg,res))
    return "", chat_history1
def clear_fn1():
    global chat_history1
    chat_history1 = []
    global chat_history1_chat_with_docs
    chat_history1_chat_with_docs = []
    clear_db()
    return "", "", chat_history1, None
def url_fn1(url_link):
    if url_link == "":
        chat_history1.append(("No URL is found", "Upload URL"))
    else:
        report = load_web(url_link)
        chat_history1.append(("URL Uploaded",report))
    return url_link,chat_history1

# MULTIMODAL RAG
from langchain_core.messages import SystemMessage
chat_history3_rag = [
    SystemMessage(content="You are a helpful assistant")
]
chat_history3 = []
def msg_input3(msg,img_path):
    res, img_res_path = continual_chat_rag(msg,img_path,chat_history3_rag)
    if img_path is not None:
        chat_history3.append((gr.Image(img_path), None))
        if img_res_path is None:
            chat_history3.append((msg,res))
            chat_history3_temp = list(chat_history3)
            chat_history3[-2] = ("USER_IMAGE", None)
        else:
            chat_history3.append((msg, gr.Image(img_res_path)))
            chat_history3.append((None,res))
            chat_history3_temp = list(chat_history3)
            chat_history3[-3] = ("USER_IMAGE", None)
            chat_history3[-2] = (msg,"AI_IMAGE")
    else:
        if img_res_path is None:
            chat_history3.append((msg,res))
            chat_history3_temp = list(chat_history3)
        else:
            chat_history3.append((msg, gr.Image(img_res_path)))
            chat_history3.append((None,res))
            chat_history3_temp = list(chat_history3)
            chat_history3[-2] = (msg,"AI_IMAGE")
    return "", chat_history3_temp, None
def clear_fn3():
    global chat_history3
    chat_history3 = []
    global chat_history3_rag
    chat_history3_rag = [
    SystemMessage(content="You are a helpful assistant")
    ] 
    return "", chat_history3, None
with gr.Blocks() as demo:

    with gr.Tab("Chat with Docs"):
        chatbot1 = gr.Chatbot(show_label=False)
        text_input1 = gr.Textbox(show_label=False, placeholder= "Enter your query:")
        with gr.Row():
            with gr.Column():
                file_box1 = gr.Files(label='PDF and Text Files Only')
                upload_button1 = gr.Button("Upload Files")
            with gr.Column():
                url_box1 = gr.Textbox(label="URL")
                clear_button1 = gr.Button("Clear All")
        upload_button1.click(fn=load1,inputs=[file_box1],outputs=[chatbot1])
        text_input1.submit(fn=msg_input1,inputs=[text_input1],outputs=[text_input1,chatbot1])
        clear_button1.click(fn=clear_fn1,outputs=[text_input1,url_box1,chatbot1,file_box1])
        url_box1.submit(fn=url_fn1,inputs=[url_box1],outputs=[url_box1,chatbot1])

    with gr.Tab("InstiGPT"):
        chatbot2 = gr.Chatbot(show_label=False)
        text_input2 = gr.Textbox(show_label=False,placeholder= "Enter your query:")
        clear_button2 = gr.Button("Clear")
        text_input2.submit(fn=msg_input2,inputs=[text_input2],outputs=[text_input2,chatbot2])
        clear_button2.click(fn=clear_fn2,outputs=[text_input2,chatbot2])

    with gr.Tab("Multimodal RAG Chat"):
        chatbot3 = gr.Chatbot(show_label=False)
        with gr.Row():
            with gr.Column():
                text_input3 = gr.Textbox(show_label=False,placeholder="Enter your query:")
                clear_button3 = gr.Button("Clear")
            image_upload3 = gr.Image(type='filepath')
        text_input3.submit(fn=msg_input3, inputs=[text_input3,image_upload3],outputs=[text_input3,chatbot3,image_upload3])
        clear_button3.click(fn=clear_fn3,outputs=[text_input3,chatbot3,image_upload3])
demo.launch()
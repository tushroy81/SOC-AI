from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import gradio as gr
load_dotenv()
# import os
# from huggingface_hub import login
# token = os.getenv('HF_TOKEN')
# login(token = token,add_to_git_credential=True)

model1 = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
model2 = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
model3 = ChatAnthropic(model = 'claude-3-opus-20240229')
model4 = ChatAnthropic(model = 'claude-3-sonnet-20240229')
model5 = ChatAnthropic(model = 'claude-3-haiku-20240307')
model6 = ChatAnthropic(model = 'claude-3-5-sonnet-20240620')

llm7 = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
model7 = ChatHuggingFace(llm=llm7)

llm8 = HuggingFaceEndpoint(
    repo_id="google/gemma-1.1-2b-it",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
model8 = ChatHuggingFace(llm=llm8) # no system

llm9 = HuggingFaceEndpoint(
    repo_id="google/gemma-1.1-7b-it",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
model9 = ChatHuggingFace(llm=llm9) # no system

model10 = ChatGoogleGenerativeAI(model = "gemini-1.0-pro") # no system

# APP WORKS

model_list = [[model1,"gemini-1.5-flash",0],[model2,"gemini-1.5-pro",0],[model3,'claude-3-opus-20240229',0],[model4,'claude-3-sonnet-20240229',0],[model5,'claude-3-haiku-20240307',0],
              [model6,'claude-3-5-sonnet-20240620',0],[model7,"mistralai/Mixtral-8x7B-Instruct-v0.1",0],[model8,"google/gemma-1.1-2b-it",0],[model9,"google/gemma-1.1-7b-it",0],
              [model10,"gemini-1.0-pro",0]]

import numpy as np
import pandas as pd

def generate_two_different_random_integers():
    random_numbers = np.random.choice(10, size=2, replace=False)
    return random_numbers[0], random_numbers[1]

a,b = generate_two_different_random_integers()

def respond(message, chat_history1:list, chat_history2:list):
    global a
    global b
    modelA = model_list[a][0]
    modelB = model_list[b][0]
    bot_message1 = modelA.invoke(message)
    bot_message2 = modelB.invoke(message)
    bot_message1 = bot_message1.content
    bot_message2 = bot_message2.content
    chat_history1.append((message, bot_message1))
    chat_history2.append((message,bot_message2))
    return "", chat_history1, chat_history2

def clear_chats():
    return "", [], []

def display_name1():
    global a
    return model_list[a][1]
def display_name2():
    global b
    return model_list[b][1]
def A_display_name1():
    global a
    return model_list[a][1] + " (VICTORY!)"
def B_display_name2():
    global b
    return model_list[b][1] + " (VICTORY!)"
def clear_name():
    return ""

a_win = False
b_win = False

def winA_fn():
    global a_win
    global b_win
    a_win = True
    b_win = False

def winB_fn():
    global a_win
    global b_win
    a_win = False
    b_win = True

def bothwin_fn():
    global a_win
    global b_win
    a_win = True
    b_win = True

def random_all():
    global a
    global b
    global a_win
    global b_win
    if (a_win == True and b_win == False):
        model_list[a][2] += 1
    if (a_win == False and b_win == True):
        model_list[b][2] += 1
    if (a_win == True and b_win == True):
        model_list[a][2] += 1
        model_list[b][2] += 1
    a_win = False
    b_win = False
    a,b= generate_two_different_random_integers()
    return "", [], []

def leaderboard_fn():
    data = {
        "Name": [model_list[i][1] for i in range(10)],
        "Score": [model_list[i][2] for i in range(10)]
    }
    df = pd.DataFrame(data)
    return df


with gr.Blocks() as demo:
    with gr.Tab("LLM Arena"):
        gr.Markdown('''# LLM Arena
                    
                    * Enter your prompt and get the answers from both models.
                    * Select the winner based on their performance (Fair Voting).
                    * Click New Round 🎲 to start a round with new models.
                    * Click Clear 🧹 to clear the old chats and start new with the same models. (Note: Clear 🧹 doesn't change the models)''')
        with gr.Row():
            with gr.Column():
                chatbot1 = gr.Chatbot(label = 'Model A')
                chatbot1_name = gr.Textbox(show_label=False,interactive=False)
            with gr.Column():
                chatbot2 = gr.Chatbot(label = "Model B")
                chatbot2_name = gr.Textbox(show_label=False,interactive=False)
            
        msg = gr.Textbox(placeholder= "Enter your prompt:",show_label=False)
        with gr.Row():
            clear = gr.Button("Clear 🧹")
            random = gr.Button("New Round 🎲")
        with gr.Row():
            winA = gr.Button("🅰️ wins")
            bothwin = gr.Button('Both 🆎 win')
            winB = gr.Button("🅱️ wins")

    with gr.Tab("Leaderboard"):
        gr.Markdown('# Leaderboard')
        leaderboard_display = gr.DataFrame(value=leaderboard_fn, headers=["Name", "Score"], interactive=False)
        update = gr.Button("Update")

    msg.submit(fn=respond, inputs=[msg, chatbot1, chatbot2], outputs=[msg, chatbot1, chatbot2])
    clear.click(clear_chats, outputs=[msg, chatbot1, chatbot2])
    random.click(random_all, outputs=[msg, chatbot1, chatbot2])
    random.click(clear_name,outputs=chatbot1_name)
    random.click(clear_name,outputs=chatbot2_name)
    winA.click(A_display_name1,outputs=chatbot1_name)
    winA.click(display_name2,outputs=chatbot2_name)
    winA.click(winA_fn)
    winB.click(display_name1,outputs=chatbot1_name)
    winB.click(B_display_name2,outputs=chatbot2_name)
    winB.click(winB_fn)
    bothwin.click(A_display_name1,outputs=chatbot1_name)
    bothwin.click(B_display_name2,outputs=chatbot2_name)
    bothwin.click(bothwin_fn)
    update.click(fn = leaderboard_fn, outputs= leaderboard_display)


demo.launch(share=True)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import gradio as gr
load_dotenv()
import os
from langchain_google_community import GoogleSearchAPIWrapper

api_key = os.getenv("GOOGLE_API_KEY_AI")
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash",google_api_key=api_key)
search = GoogleSearchAPIWrapper()

system_message1 = ("For the given human message, you need to tell if human is asking about condition(weather,climate,rain,wind,flood etc.) of new place or the previous one: yes or no only"
                   "No need to describe the weather"
                   "also reply that place after yes with single space (in case of yes)")
message2 = None
first_chat = True
documents = None
weather = None
loc0 = None
def AI_Reply(query):
    global system_message1
    message1 = [
        SystemMessage(content = system_message1),
        HumanMessage(content = query)
    ]
    response1 = llm.invoke(message1).content
    global documents
    global first_chat
    global message2
    global loc0
    global weather
    if weather is None:
        return ("Choose Weather Channel")
    
    if ('no' in response1.lower() and first_chat is True):
        return ("Please mention the location")
    
    if response1.lower()[:3] == 'yes':
        message2 = []
        first_chat = False
        place = response1[4:]
        loc0 = place[:-1]
        if weather == "The Weather Channel":
            result = search.results(f"{place} weather.com",num_results=1)
        if weather == "AccuWeather":
            result = search.results(f"{place} accuweather.com",num_results=1)
        link = result[0]['link']
        loader = WebBaseLoader(link)
        documents = loader.load()
        documents = documents[0].page_content
        message2.append(SystemMessage(content=("Answer all the human querires from the following context."
                                               "If not mentioned, give today's data only from the context"
                                               f"convert F(farhaneit) to C(celcius).\n\n{documents}")))
        message2.append(HumanMessage(content=query))
        response2 = llm.invoke(message2).content
        message2.append(AIMessage(content=response2))

        return response2
    
    if ('no' in response1.lower() and first_chat == False):
        message2.append(HumanMessage(content=query))
        response2_1 = llm.invoke(message2).content
        message2.append(AIMessage(content=response2_1))

        return response2_1

def weather_channel_update(weather_input):
    global weather
    weather = weather_input

chat_history = []

def msg_input(msg):
    res = AI_Reply(msg)
    global loc0
    chat_history.append((msg,res))
    return "", chat_history, loc0
def clear_fn():
    global chat_history
    chat_history = []
    global loc0
    loc0 = None
    global first_chat
    first_chat = True
    return "", chat_history, loc0
def location_update(loc):
    global loc0
    loc0 = loc
    msg = f"{loc} Weather"
    res = AI_Reply(msg)
    chat_history.append((msg,res))
    return loc, chat_history

with gr.Blocks() as demo:
    weather_channel =  gr.Dropdown(label="Weather Channel",choices=["The Weather Channel","AccuWeather"])
    chatbot = gr.Chatbot(show_label=False)
    text_input = gr.Textbox(show_label=False,placeholder= "Enter your query:")
    with gr.Row():
        location = gr.Textbox(label="Location")
        location.submit(fn = location_update, inputs = location, outputs=[location,chatbot])
        clear_button = gr.Button('Clear')
        clear_button.click(fn=clear_fn,outputs=[text_input,chatbot,location])

    text_input.submit(fn=msg_input,inputs=[text_input],outputs=[text_input,chatbot,location])
    weather_channel.change(fn = weather_channel_update,inputs=weather_channel)
    weather_channel.change(fn = clear_fn,outputs=[text_input,chatbot,location])

demo.launch()

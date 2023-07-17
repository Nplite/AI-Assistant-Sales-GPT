import argparse
import os
import json
import pyttsx3
import speech_recognition as sr

from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

engine = pyttsx3.init()

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    except sr.RequestError as e:
        print(f"Error occurred during speech recognition: {e}")
    return ""


def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to agent config file', default='')
    parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
    parser.add_argument('--max_num_turns', type=int, help='Maximum number of turns in the sales conversation',
                        default=10)

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns

    llm = ChatOpenAI(temperature=0.9, stop="<END_OF_TURN>")

    if config_path == '':
        print('No agent config specified, using a standard config')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f'Agent config {config}')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)

    sales_agent.seed_agent()
    print('=' * 10)
    cnt = 0
    while cnt != max_num_turns:
        cnt += 1
        if cnt == max_num_turns:
            print('Maximum number of turns reached - ending the conversation.')
            break
        sales_agent.step()

        # end conversation
        if '<END_OF_CALL>' in sales_agent.conversation_history[-1]:
            print('Sales Agent determined it is time to end the conversation.')
            break
        
        # Get user input (audio or text)
        user_input = speech_to_text()
        if not user_input:
            user_input = input("Your response: ")

        # Pass user input to the sales agent
        if user_input:
            sales_agent.human_step(user_input)
        
        print('=' * 10)
        
        # Get sales agent's response
        response = sales_agent.conversation_history[-1]
        
        # Convert response to speech and play it
        text_to_speech(response)

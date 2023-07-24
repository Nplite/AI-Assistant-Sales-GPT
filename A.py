import argparse
import os
import json
from flask import Flask, render_template, request
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Initialize Flask app
app = Flask(__name__)

parser = argparse.ArgumentParser(description='Description of your program')

# Add arguments
parser.add_argument('--config', type=str, help='Path to agent config file', default='')
parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
parser.add_argument('--max_num_turns', type=int, help='Maximum number of turns in the sales conversation', default=10)

# Parse arguments
args = parser.parse_args()

# Access arguments
config_path = args.config
verbose = args.verbose
max_num_turns = args.max_num_turns

llm = ChatOpenAI(temperature=0.9, stop = "<END_OF_TURN>")

if config_path=='':
    print('No agent config specified, using a standard config')
    sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
else:
    with open(config_path,'r') as f:
        config = json.load(f)
    print(f'Agent config {config}')
    sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)

sales_agent.seed_agent()

# # Function to generate response with SalesGPT
# def generate_response_with_gpt(user_input):
#     response = sales_agent.human_step(user_input)
#     if response is not None:
#         return response.get('response')
#     else:
#         return 'Sorry, I could not generate a response at the moment.'



@app.route('/')
def index():
    return render_template('first.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    # cnt = 0
    # while cnt !=max_num_turns:
    #     cnt+=1
    #     if cnt==max_num_turns:
    #         print('Maximum number of turns reached - ending the conversation.')
    #         break
    #     sales_agent.step()

    #     # end conversation 
    #     if '<END_OF_CALL>' in sales_agent.conversation_history[-1]:
    #         print('Sales Agent determined it is time to end the conversation.')
    #         break



    if 'user_input' in request.form:
        user_input = request.form['user_input']
        # response1 = sales_agent.human_step(user_input)
        response1 = "Hi"
    else:
        print('Data not passed')




    # user_input = request.form['user_input']
    # response = sales_agent.conversation_history[0]




    return render_template('first.html', response = response1)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import streamlit as st
import cohere 
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import warnings
warnings.filterwarnings('ignore')
# initialize the Cohere Client with an API Key
# api_key = st.text_input("Enter your API key", value="", type="password")
api_key = 'iBAdfWjnHxBfG5fr4jpVfVkYYxLssmxglN8s2Oc3'
# co = cohere.Client(api_key=api_key)
co = cohere.Client(api_key)

def execution(input):
    bot = ChatBot("My Bot")
    list_trainer = ListTrainer(bot)
    corpus_trainer = ChatterBotCorpusTrainer(bot)

    # to train all
    # corpus_trainer.train("chatterbot.corpus.english", "chatterbot.corpus.hindi")
    corpus_trainer.train("chatterbot.corpus.english", "chatterbot.corpus.hindi")

    result = bot.get_response(input)
    return st.write(f"Chatbot: {result}")

# title of the chatbot 
st.title("Welcome to Our chatbot")

input = st.text_input('User:')
print("input here:", input)
execution(input=input)

import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

st.cache(suppress_st_warning=True)
@st.experimental_singleton

def execution(input):
    bot = ChatBot("My Bot")
    list_trainer = ListTrainer(bot)
    corpus_trainer = ChatterBotCorpusTrainer(bot)

    # to train all
    # corpus_trainer.train("chatterbot.corpus.english", "chatterbot.corpus.hindi")
    corpus_trainer.train("chatterbot.corpus.english", "chatterbot.corpus.hindi")

    result = bot.get_response(input)
    return st.write(f"Chatbot: {result}")


st.title("Welcome to Our chatbot")
input = st.text_input('User:')
print("input here:", input)
execution(input=input)
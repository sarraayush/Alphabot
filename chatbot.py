import streamlit as st
import cohere 
import numpy as np
import pandas as pd
from tqdm import tqdm
import umap
# import altair as alt
from chatterbot import ChatBot
from annoy import AnnoyIndex
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import warnings
warnings.filterwarnings('ignore')
# initialize the Cohere Client with an API Key
# api_key = st.text_input("Enter your API key", value="", type="password")
api_key = 'iBAdfWjnHxBfG5fr4jpVfVkYYxLssmxglN8s2Oc3'
# co = cohere.Client(api_key=api_key)
co = cohere.Client(api_key)

embeds = co.embed(texts=list(df['text']),
                  model='large',
                  truncate='LEFT').embeddings

search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(10) # 10 trees
search_index.save('test.ann')

# Choose an example (we'll retrieve others similar to it)
example_id = 92
# Retrieve nearest neighbors
similar_item_ids = search_index.get_nns_by_item(example_id,10,
                                                include_distances=True)
# Format and print the text and distances
results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                             'distance': similar_item_ids[1]}).drop(example_id)
print(f"Question:'{df.iloc[example_id]['text']}'\nNearest neighbors:")

reducer = umap.UMAP(n_neighbors=20) 
umap_embeds = reducer.fit_transform(embeds)
# Prepare the data to plot and interactive visualization
# using Altair
df_explore = pd.DataFrame(data={'text': df['text']})
df_explore['x'] = umap_embeds[:,0]
df_explore['y'] = umap_embeds[:,1]

# Plot
chart = alt.Chart(df_explore).mark_circle(size=60).encode(
    x=#'x',
    alt.X('x',
        scale=alt.Scale(zero=False)
    ),
    y=
    alt.Y('y',
        scale=alt.Scale(zero=False)
    ),
    tooltip=['text']
).properties(
    width=700,
    height=400
)
chart.interactive()
                             
#st.cache(suppress_st_warning=True)
#@st.experimental_singleton

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

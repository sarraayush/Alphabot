import cohere as cohere
#!pip install cohere umap-learn altair annoy tqdmimport cohere

import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
# ---------------------
#import requests
import csv
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
# ---------------------
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

api_key = 'iBAdfWjnHxBfG5fr4jpVfVkYYxLssmxglN8s2Oc3'

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key=api_key)

#print(df)

industry = input('Enter an industry: ')

df = pd.read_csv('financials.csv')

sector = []
company_name = []
sales = []
earnings = []
market_cap = []

competitors = []
competitor_sales_records = []
competitor_earnings_records = []
competitor_market_records = []

company_website = []
company_data = []

sector.append(df['Sector'])
company_name.append(df['Name'])
sales.append(df['Price/Sales'])
earnings.append(df['Price/Earnings'])
market_cap.append(df['Market Cap'])

sector_len = len(sector)

# getting a list of top competitors
for i in range(len(df)):
  if industry == df.loc[i, "Sector"]:

    #index = sector.index(industry)
    if df.loc[i, "Name"] and df.loc[i, "Price/Sales"] and df.loc[i, "Market Cap"] == '':
        break

    else:
        competitor = df.loc[i, "Name"]
        sales_records = df.loc[i, "Price/Sales"]
        #earnings_records = earnings[i]
        market_records = df.loc[i, "Market Cap"]

    competitors.append(competitor)
    competitor_sales_records.append(sales_records)
    #competitor_earnings_records.append(earnings_records)
    competitor_market_records.append(market_records)



# removing blank values from the lists
#filter(None, competitor_sales_records)
#filter(None, competitor_earnings_records)
#filter(None, competitor_market_records)
#filter(None, competitors)

# converting the list values to float for the graphs
competitor_sales_records = list(np.float_(competitor_sales_records))
#competitor_earnings_records = list(np.float_(competitor_earnings_records))
competitor_market_records = list(np.float_(competitor_market_records))

# getting only top 20 numbers from the list --> so only top 20 companies
dict_competitor_sales_records = dict(zip(competitors, competitor_sales_records))
sorted(dict_competitor_sales_records)
dict_competitor_sales_records = dict(itertools.islice(dict_competitor_sales_records.items(), 20))
competitors = dict_competitor_sales_records.keys()
competitor_sales_records = dict_competitor_sales_records.values()

#dict_competitor_earnings_records = dict(zip(competitors, competitor_earnings_records))
#sorted(dict_competitor_earnings_records)
#dict_competitor_earnings_records = dict(itertools.islice(dict_competitor_earnings_records.items(), 20))
#competitors = dict_competitor_earnings_records.keys()
#competitor_earnings_records = dict_competitor_earnings_records.values()

dict_competitor_market_records = dict(zip(competitors, competitor_market_records))
sorted(dict_competitor_market_records)
dict_competitor_market_records = dict(itertools.islice(dict_competitor_market_records.items(), 20))
competitors = dict_competitor_market_records.keys()
competitor_market_records = dict_competitor_market_records.values()


# creating 3 pie charts based on the data from lists

fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

labels = list(competitors)

ax1.pie(competitor_sales_records, autopct='%1.1f%%', labels = labels, textprops ={"fontsize":5})
#ax2.pie(competitor_earnings_records,
#        labels=labels,
#        autopct='%0.9f%%',
#        shadow=True,
#        startangle=90)
ax3.pie(competitor_market_records, autopct='%1.1f%%', labels = labels, textprops ={"fontsize":5})

ax1.axis('equal')
#ax2.axis('equal')
ax3.axis('equal')

ax1.set_title('Sales Records for top competitors in ' + industry)
#ax2.set_title('Earnings Records for top competitors')
ax3.set_title('Market Cap Records for top competitors in ' + industry)

#plt.legend()
plt.show()

#print(competitors)

# code for finding company website from company name

#for competitor in competitors:
#ind = competitors.index(competitor) # index for collecting other data related to the competitor
#response = requests.get(f'https://autocomplete.clearbit.com/v1/companies/suggest?query={competitor}')

#data = response.json()
#if len(data) > 0:
#domain = data[0]['domain']
#print(domain)
#company_website.append(domain)

#print("Status Code", response.status_code)

#print("Company Data ", company_website)



# Get the embeddings
embeds = co.embed(texts=list(df['Name']),
                  model="large",
                  truncate="LEFT").embeddings

# Check the dimensions of the embeddings
embeds = np.array(embeds)
embeds.shape

# Create the search index, pass the size of embedding
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
results = pd.DataFrame(data={'Name': df.iloc[similar_item_ids[0]]['Name'],
                             'Earnings/Share': similar_item_ids[1]}).drop(example_id)

print(f"Name:'{df.iloc[example_id]['Earnings/Share']}'\nNearest neighbors:")
results

query = industry

# Get the query's embedding
query_embed = co.embed(texts=[query],
                  model="large",
                  truncate="LEFT").embeddings

# Retrieve the nearest neighbors
similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                include_distances=True)
# Format the results
results = pd.DataFrame(data={'Name': df.iloc[similar_item_ids[0]]['Name'],
                             'Earnings/Share': similar_item_ids[1]})


print(f"Query:'{query}'\nNearest neighbors:")
results

# UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
reducer = umap.UMAP(n_neighbors=20)
umap_embeds = reducer.fit_transform(embeds)
# Prepare the data to plot and interactive visualization
# using Altair
df_explore = pd.DataFrame(data={'Name': df['Name']})
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
    tooltip=['Name']
).properties(
    width=700,
    height=400
)
chart.interactive()
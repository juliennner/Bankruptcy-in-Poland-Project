import gzip
import json

import pandas as pd
import wqet_grader
from IPython.display import VimeoVideo



# In the terminal window, locate the data file for this project and decompress it.

get_ipython().run_cell_magic('bash', '', 'cd data\ngzip -dkf poland-bankruptcy-data-2009.json.gz')


# ## Explore


# Load the data into a DataFrame.
df = ...
df.head()


# Open file and load JSON
with open("data/poland-bankruptcy-data-2009.json","r") as read_file:
    poland_data = json.load(read_file)

print(type(poland_data))

# poland_data.keys()
poland_data["data"][0]



VimeoVideo("693794783", h="8d333027cc", width=600)


# Calculate number of features
len(poland_data["data"][0])


# Iterate through companies
for item in poland_data["data"]:
    if len(item) != 66:
        print("Alert!!!")

# Open compressed file and load contents
with gzip.open("data/poland-bankruptcy-data-2009.json.gz","r") as read_file:
    poland_data_gz = json.load(read_file)

print(type(poland_data_gz))

# Explore `poland_data_gz`
print(poland_data_gz.keys())
print(len(poland_data_gz["data"]))
print(len(poland_data_gz["data"][0]))


df = pd.DataFrame().from_dict(poland_data_gz["data"]).set_index("company_id")
print(df.shape)
df.head()


def wrangle(filename):
    with gzip.open(filename,"r") as f:
        data = json.load(f)
    df = pd.DataFrame().from_dict(data["data"]).set_index("company_id")
    return df


# In[22]:


df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()

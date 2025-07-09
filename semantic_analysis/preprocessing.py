import pandas as pd
import numpy as np 
import ast


### PRE-PROCESS PROMPT
def preprocess(text):
    text = str(text).lower().strip()
    return text 

df = pd.read_csv('datasets/combined_modified_csv.csv')

df['Prompt'] = df['Prompt'].apply(preprocess)

#print(df['Prompt'].iloc[1])
#print(df['embedding'].iloc[1])

def parse_embedding(s):
    # Remove brackets and extra spaces
    s = s.replace('[', '').replace(']', '')
    return np.fromstring(s, sep=' ')

# convert string "embedding" into a list of floats with comma in-between
df['embedding'] = df['embedding'].apply(parse_embedding)
embeddings = np.stack(df['embedding'].values) #convert into 2D-array for CLASSIFICATION

# embeddings: input & df['label']: output
# stratify=df['label']: Ensures proportion of each class (benign/malicious) is same in both training and testing sets.
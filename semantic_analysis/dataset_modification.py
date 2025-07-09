import numpy as np 
import pandas as pd 

print('\n', '### BENIGN DATABASE WITH LABEL 0: HARMLESS')
benign_db = pd.read_csv('datasets/benign_deepset.csv') #label 0: harmless
print(benign_db.head())

print('\n', '### INJECTION DATABASE WITH LABEL 1: HARMFUL')
injection_db = pd.read_csv('datasets/malicous_deepset.csv') #label 1: harmful
print(injection_db.head())

print('\n', '### BENIGN & INJECTION DATABASE COMBINED WITH LABELS 0 & 1: HARMLESS & HARMFUL')

# Reset index for both, but keep the original as a column if needed
benign_db = benign_db.reset_index(drop=True)
injection_db = injection_db.reset_index(drop=True)

# Update the index of the malicious_db to continue from the last index of benign_db
injection_db.index = injection_db.index + len(benign_db)

# Concatenate: benign first, then injection
combined_db = pd.concat([benign_db, injection_db])

# Reset the index column name for a clean CSV
combined_db.index.name = 'index'

# Save to new CSV
combined_db.to_csv('datasets/combined_benign_malicious.csv')

print('Combined CSV saved as datasets/combined_benign_malicious.csv')


### DROPPING REDUNDANT "INDEX" COLUMNS

db = pd.read_csv('datasets/combined_benign_malicious.csv')
db_modified = db.drop(['Unnamed: 0', 'idx'], axis=1) #column: axis=1
db_modified.to_csv('datasets/combined_modified_csv.csv')

print('Modified CSV saved as datasets/combined_modified_csv.csv')

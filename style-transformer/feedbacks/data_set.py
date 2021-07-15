import pandas as pd
import csv

df = pd.read_csv("previous_two_context_full.csv")

def strip_delimiter(text):
    return text.replace('__p1__', ' ').replace('__p2__', ' ').strip()

df['previous_2_context_clean'] = df['previous_2_context'].apply(strip_delimiter)
df["context_response"] =  df['previous_2_context_clean'] + [' '] + df['response'] + ['__label__'] + df['label'].astype('str')
s= df["context_response"][2]
print(s.split('__label__'))

df["context_response"][:52148].to_csv('train.source', header=False, index=False,quoting=csv.QUOTE_NONNUMERIC)
df["context_response"][52148:62148].to_csv('val.source', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
df["context_response"][62148:].to_csv('test.source', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
df["response"][:52148].to_csv('train.target', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
df["response"][52148:62148].to_csv('val.target', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
df["response"][62148:].to_csv('test.target', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)

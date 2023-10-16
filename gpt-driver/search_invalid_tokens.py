import pickle
import json
import ast
import numpy as np

filename = "outputs/gpt_uniad.pkl"
data_dict = pickle.load(open("outputs/gpt_uniad.pkl", "rb"))

split = json.load(open('data/split.json', 'r'))

train_tokens = split["train"]
test_tokens = split["val"]

untest_tokens = [
]

for token in test_tokens:
    if token not in data_dict:
        untest_tokens.append(token)

print("#### Invalid Tokens ####")
for token in untest_tokens:
    print(token)

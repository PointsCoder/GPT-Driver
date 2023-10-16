import openai
import pickle
import json
import ast
import numpy as np
import time
import argparse
from prompt_message import system_message, generate_user_message, generate_assistant_message
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

parser = argparse.ArgumentParser(description="GPT-Driver test.")
parser.add_argument("-i", "--id", type=str, help="GPT model id")
parser.add_argument("-o", "--output", type=str, help="output file name")
args = parser.parse_args()

saved_traj_name = "outputs/" + args.output + ".pkl"
saved_text_name = "outputs/" + args.output + "_text.pkl"
temp_text_name = "outputs/" + args.output + "_temp.jsonl"

openai.api_key = "" # insert your API key here

data = pickle.load(open('data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('data/split.json', 'r'))

train_tokens = split["train"]
test_tokens = split["val"]

text_dict, traj_dict = {}, {}

invalid_tokens = []

untest_tokens = [
]

for token in test_tokens:
    if len(untest_tokens) > 0 and token not in untest_tokens: 
        continue

    print()
    print(token)

    time.sleep(1)    
    user_message = generate_user_message(data, token)
    assitant_message = generate_assistant_message(data, token)
    model_id = args.id
    completion = completion_with_backoff(
        model=model_id,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )
    # import pdb; pdb.set_trace()
    result = completion.choices[0].message["content"]
    print(f"GPT  Planner:\n {result}")
    print(f"Ground Truth:\n {assitant_message}")
    output_dict = {
        "token": token,
        "GPT": result,
        "GT": assitant_message, 
    }

    text_dict[token] = result

    traj = result.split("\n")[-1]
    try:
        traj = ast.literal_eval(traj)
        traj = np.array(traj)
    except:
        print(f"Invalid token: {token}")
        invalid_tokens.append(token)
        continue
    traj_dict[token] = traj

    with open(temp_text_name, "a+") as file:
        file.write(json.dumps(output_dict) + '\n')

    # output_dicts = []
    # with open(temp_text_name, "r") as file:
    #     for line in file:
    #         output_dicts.append(json.loads(line))

    if len(untest_tokens) > 0:
        exist_dict = pickle.load(open(saved_traj_name, 'rb'))
        exist_dict.update(traj_dict)
        fd = open(saved_traj_name, "wb")
        pickle.dump(exist_dict, fd)

print("#### Invalid Tokens ####")
for token in invalid_tokens:
    print(token)

if len(untest_tokens) == 0:
    with open(saved_text_name, "wb") as f:
        pickle.dump(text_dict, f)
    with open(saved_traj_name, "wb") as f:
        pickle.dump(traj_dict, f)
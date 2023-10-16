import pickle
import json
import ast
import numpy as np

filename = "outputs/gpt_incontext_temp.jsonl"
data_dict = {}
with open(filename, 'r') as file:
    for line in file:
        json_obj = json.loads(line.strip())
        token = json_obj['token']
        try:
            gpt_text = json_obj['GPT']
            traj = gpt_text.split("\n")[-1]
            traj = ast.literal_eval(traj)
            traj = np.array(traj)
            if traj.shape[0] != 6 or traj.shape[1] != 2:
                print(f"Invalid token: {token}")
                continue
            data_dict[token] = traj
        except:
            print(f"Invalid token: {token}")
            continue

with open("outputs/gpt_incontext.pkl", "wb") as f:
    pickle.dump(data_dict, f)

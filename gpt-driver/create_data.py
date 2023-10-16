import pickle
import ndjson
import json
import tiktoken
from prompt_message import system_message, generate_user_message, generate_assistant_message

data = pickle.load(open('data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('data/split.json', 'r'))

train_tokens = split["train"]
val_tokens = split["val"]
num_train_samples = len(train_tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

num_language_tokens = 0
num_system_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0

traj_only = False

train_messages = []
for token_i, token in enumerate(train_tokens):
    if token_i >= train_ratio * num_train_samples:
        break 
    user_message = generate_user_message(data, token)
    assitant_message = generate_assistant_message(data, token, traj_only=traj_only)
    if len(assitant_message.split("\n")) > 6:
        print()
        print(token)
        print(system_message)
        print(user_message)
        print(assitant_message)
    num_language_tokens += len(encoding.encode(system_message))
    num_system_tokens += len(encoding.encode(system_message))
    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))


    train_message = {"messages": 
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}, 
            {"role": "assistant", "content": assitant_message}
        ]
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of system tokens: {num_system_tokens}")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")

with open("data/train.json", "w") as f:
    ndjson.dump(train_messages, f)
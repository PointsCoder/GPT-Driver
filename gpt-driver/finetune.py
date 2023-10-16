import openai

openai.api_key = "" # insert your API key here

response = openai.File.create(file=open("train.json", "r"), purpose='fine-tune')
print(response)
train_file_id = response["id"]

response = openai.FineTuningJob.create(training_file=train_file_id, model="gpt-3.5-turbo", hyperparameters={"n_epochs":1, })
print(response)
finetune_job_id = response["id"]

# List 10 fine-tuning jobs
openai.FineTuningJob.list(limit=10)

# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve(finetune_job_id)

# List up to 10 events from a fine-tuning job
openai.FineTuningJob.list_events(id=finetune_job_id, limit=10)
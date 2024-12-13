!pip install transformers
!pip install datasets
!pip install rouge_score
!pip install rouge
!pip install nltk
!pip install peft
!pip install bitsandbytes

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, BitsAndBytesConfig
from rouge import Rouge
from itertools import islice
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from google.colab import drive

import numpy as np
import torch
import psutil
import time

dataset = load_dataset("knkarthick/dialogsum", trust_remote_code=True)

print(dataset['train'])
print(len(dataset['train']))

print(dataset['train'][0])
print()
print(dataset['train'][0]['dialogue'])
print()
print(dataset['train'][0]['summary'])

model_name='google/flan-t5-large'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,                                                                                       
                                              device_map="auto",
                                              quantization_config=BitsAndBytesConfig(
                                                  load_in_4bit=True,
                                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                                  bnb_4bit_quant_type="nf4",
                                              ),
                                              torch_dtype=torch.bfloat16,)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

device = torch.device("cuda")
model.to(device)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

def get_memory_usage():
    process = psutil.Process()
    ram_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    return ram_usage

# Initialize timers and counters
batch_times = []
start_time = time.time()

# Measure initial memory usage
initial_ram_usage = get_memory_usage()
initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)

drive.mount('/content/drive')
output_dir = '/content/drive/MyDrive/training-run1'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    bf16=True
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()

model_path = '/content/drive/MyDrive/training-run1/checkpoint-1558'

trained = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
device = torch.device("cuda")
trained.to(device)

batched_summaries=[]
example_indices = range(len(dataset['train']))  # In the end replace it with len(dataset['train'])
batch_size = 32

# Function to yield batches
def batched_indices(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield [first] + list(islice(iterator, size - 1))

# Process each batch of indices
for batch_num, batch in enumerate(batched_indices(example_indices, batch_size)):
    print(batch_num + 1, end=" ")

    # Collect dialogues and summaries for the batch
    dialogues = [dataset['train'][index]['dialogue'] for index in batch]
    summaries = [dataset['train'][index]['summary'] for index in batch]

    # Tokenize all dialogues in the batch simultaneously
    inputs = tokenizer(dialogues, return_tensors='pt', padding=True, truncation=True).to(device)
    # inputs = tokenizer(dialogues, return_tensors='pt', padding=True, truncation=True)

    # Generate summaries for all dialogues in the batch at once
    outputs = trained.generate(inputs["input_ids"], max_new_tokens=50)

    # Decode and print each example in the batch
    for i, (dialogue, summary, output) in enumerate(zip(dialogues, summaries, outputs)):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        batched_summaries.append(decoded_output)
        
for i in batched_summaries:
    print(i)

# Total number of examples and batch size
total_examples = 10 #in the final one replace this with the total size of the training set
batch_size = 8
dash_line = '-' * 100
zero_shot_summaries_batch = []

# Generate example indices for the full dataset
example_indices = range(total_examples)

# Process each batch of indices
for batch_num, batch in enumerate(batched_indices(example_indices, batch_size)):
    print(f'{batch_num + 1}')

    # Collect dialogues and summaries for the batch
    dialogues = [dataset['train'][index]['dialogue'] for index in batch]
    summaries = [dataset['train'][index]['summary'] for index in batch]

    # Construct prompts for each dialogue in the batch
    prompts = [f"""
      Summarize the following conversation.

      {dialogue}

      Summary:
    """ for dialogue in dialogues]

    # Tokenize all prompts in the batch simultaneously
    # inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate summaries for all prompts in the batch at once
    outputs = trained.generate(inputs["input_ids"], max_new_tokens=50)

    # Decode and print each example in the batch
    for i, (prompt, summary, output) in enumerate(zip(prompts, summaries, outputs)):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        zero_shot_summaries_batch.append(decoded_output)

    # Optional: Break after a certain number of batches for testing (remove or comment for full run)
    # if batch_num >= some_value:
    #     break
for i in zero_shot_summaries_batch:
    print(i)

# Total number of examples and batch size
total_examples = len(dataset['train']) #in the final one replace this with the total size of the training set
batch_size = 32
dash_line = '-' * 100
zero_shot_changed_summaries_batch = []

# Generate example indices for the full dataset
example_indices = range(total_examples)

# Process each batch of indices
for batch_num, batch in enumerate(batched_indices(example_indices, batch_size)):
    batch_start_time = time.time()
    print(batch_num + 1, end=" ")

    # Collect dialogues and summaries for the batch
    dialogues = [dataset['train'][index]['dialogue'] for index in batch]
    summaries = [dataset['train'][index]['summary'] for index in batch]

    # Construct prompts for each dialogue in the batch
    prompts = [f"""
      Dialogue

      {dialogue}

      What was going on?
    """ for dialogue in dialogues]

    # Tokenize all prompts in the batch simultaneously
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate summaries for all prompts in the batch at once
    outputs = trained.generate(inputs["input_ids"], max_new_tokens=100)

    # Decode and append summaries
    for i, (prompt, summary, output) in enumerate(zip(prompts, summaries, outputs)):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        zero_shot_changed_summaries_batch.append(decoded_output)

    # Record batch processing time
    batch_times.append(time.time() - batch_start_time)

# Measure final memory usage
final_ram_usage = get_memory_usage()
final_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
total_time = time.time() - start_time

# Calculate metrics
average_batch_time = sum(batch_times) / len(batch_times)

print(f"Initial RAM Usage (MB): {initial_ram_usage:.2f}")
print(f"Final RAM Usage (MB): {final_ram_usage:.2f}")
print(f"Initial GPU Memory Usage (MB): {initial_gpu_memory:.2f}")
print(f"Final GPU Memory Usage (MB): {final_gpu_memory:.2f}")
print(f"Average Inference Time per Batch (s): {average_batch_time:.2f}")
print(f"Total Inference Time (s): {total_time:.2f}")

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Lists to store the scores
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Calculate ROUGE scores for each pair of predicted and reference summaries
for pred, ref in zip(zero_shot_changed_summaries_batch, dataset['train']['summary']):
    scores = scorer.score(ref, pred)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

# Calculate the average ROUGE scores across all pairs
avg_rouge1 = np.mean(rouge1_scores)
avg_rouge2 = np.mean(rouge2_scores)
avg_rougeL = np.mean(rougeL_scores)

# Display results
print("Average ROUGE-1 F1 Score:", avg_rouge1)
print("Average ROUGE-2 F1 Score:", avg_rouge2)
print("Average ROUGE-L F1 Score:", avg_rougeL)

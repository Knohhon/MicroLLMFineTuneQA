import collections
from functools import partial

import evaluate
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import ModernBertForQuestionAnswering, AutoTokenizer
import os
import torch
import csv
from squad_dataset_preprocessing import preprocess_train_examples, preprocess_valid_examples
from compute_train_metrics import compute_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '../models/answerdotai--ModernBERT-base'
data = load_dataset("squad")

tokenizer = AutoTokenizer.from_pretrained(
    model_path, clean_up_tokenization_spaces=False)


preprocess_train_data = partial(
    preprocess_train_examples, tokenizer=tokenizer, max_length=384, stride=128)
processed_train_data = data["train"].map(preprocess_train_data, batched=True, remove_columns=data["train"].column_names)

preprocess_valid_data = partial(
    preprocess_valid_examples, tokenizer=tokenizer, max_length=384, stride=128)
processed_valid_data = data["validation"].map(preprocess_valid_data, batched=True, remove_columns=data["validation"].column_names)

model = ModernBertForQuestionAnswering.from_pretrained(model_path, attn_implementation="flash_attention_2", dtype=torch.bfloat16).to(device)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3
)

training_args = TrainingArguments(
    output_dir='../checkpoints',
    logging_dir='../logs',
    eval_strategy="steps",
    logging_steps=100,
    logging_strategy="steps",
    save_steps=1000,
    save_strategy="steps",
    learning_rate=5e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    #warmup_ratio=0.6,
    lr_scheduler_type='linear',
    bf16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_data,
    eval_dataset=processed_valid_data,
    processing_class=tokenizer,
    #callbacks=[lc_callback],
    callbacks=[early_stopping_callback]
)

predictions, _, _ = trainer.predict(processed_valid_data)
start_logits, end_logits = predictions
base_metrics = compute_metrics(start_logits, end_logits, processed_valid_data, data["validation"])
print(f'Baseline ModernBERT metrics with SQuAD dataset: {base_metrics}')

trainer.train()

predictions, _, _ = trainer.predict(processed_valid_data)
start_logits, end_logits = predictions
post_sft_metrics = compute_metrics(start_logits, end_logits, processed_valid_data, data["validation"])
print(f'Final ModernBERT metrics with SQuAD dataset: {post_sft_metrics}')

print('Save metrics...')
trainer.save_metrics('/MicroLLMFineTuneQA/metrics/sqaud_fine_tune_metric.json')
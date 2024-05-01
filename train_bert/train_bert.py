import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/data.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_weights = (1 - (df['labels'].value_counts().sort_index() / len(df))).values
class_weights = torch.from_numpy(class_weights).float().to(device)

def get_dataset(csv_path, test_size=0.2):
    full_dataset = load_dataset('csv', data_files=csv_path)
    full_dataset = full_dataset.filter(lambda example: example['text'] is not None and example['labels'] is not None)
    full_dataset = full_dataset.class_encode_column("labels")
    dataset = full_dataset['train'].train_test_split(test_size=test_size, stratify_by_column='labels')
    return dataset
dataset = get_dataset('data/data.csv')
labels = sorted(df['labels'].value_counts().keys())

id2label = {}
label2id = {}
for i, label in enumerate(labels):
    id2label[i] = label
    label2id[label] = i

model_name = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

model.to(device)

from transformers import AdamW, get_scheduler

dataset_len = (dataset['train'].num_rows + dataset['test'].num_rows)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * dataset_len


lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps= int(0.1 * num_epochs * dataset_len),
    num_training_steps=num_training_steps
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


f1_metric = evaluate.load('f1')

training_args = TrainingArguments(
    output_dir="./results/multiclass-rubert/",
    num_train_epochs=50,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    warmup_steps=200,
    weight_decay=0.01,
    logging_strategy='no',
    evaluation_strategy="steps",
    eval_steps=50,
    save_only_model=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    metric_for_best_model='f1',
    greater_is_better=True,
    eval_accumulation_steps=32,
    fp16=True,  # mixed precision
    load_best_model_at_end = True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels, average='macro')

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=[optimizer, lr_scheduler],
)

trainer.train()


from tqdm import tqdm


y_pred = []
y_true = tokenized_dataset["test"]['labels']
with torch.no_grad():
    for i in tqdm(range(len(tokenized_dataset["test"]))):
        logits = model(**tokenizer(tokenized_dataset["test"][i]['text'], max_length=512, truncation=True, return_tensors="pt").to(device))
    
        predicted_class_id = logits.logits.argmax().item()
        # model.config.id2label[predicted_class_id]
    
        y_pred.append(predicted_class_id)

print(f1_score(y_true, y_pred, average='macro'))
print(recall_score(y_true, y_pred, average='macro'))
print(precision_score(y_true, y_pred, average='macro'))

print(classification_report(y_true, y_pred))

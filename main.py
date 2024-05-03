import pandas as pd
import numpy as np
from typing import List
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import re
from fastapi import FastAPI
from pydantic import BaseModel

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ru_rag.serve import populate_db, find_similar
from ru_rag.serve import answer

# from ru_rag.utils import download_llama_model
import os
from dotenv import load_dotenv, find_dotenv


class InputData(BaseModel):
    text: str


# Определяем модель данных для выходных параметров
class OutputData(BaseModel):
    prediction: str


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def create_embeddings(texts: List[str], model: AutoModel, tokenizer: AutoTokenizer, device: torch.device) -> Tensor:
    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_inputs = tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors='pt')
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            model_output = model(**encoded_inputs)
        batch_embeddings = average_pool(model_output.last_hidden_state, encoded_inputs['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
device = torch.device('cuda:0')

df = pd.read_csv("data/row/train_dataset.csv",sep=',')


model.to(device)
passage_embeddings = create_embeddings(texts=df["QUESTION"].to_list(),model=model,tokenizer=tokenizer,device=device)

app = FastAPI()


@app.post("/new_find_similar", response_model=OutputData)
def predict_new_find_similar(input_data: InputData):
    client_request = input_data.text
    query_text = f"query: {client_request}"

    query_batch_dict = tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
    query_outputs = model(**query_batch_dict.to(device))
    query_embedding = average_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    scores = (query_embedding @ passage_embeddings.T) * 100

    scores = scores[0].cpu().detach().numpy()
    top_3_results = np.argsort(scores)[-4:][::-1] if len(scores) >= 4 else np.argsort(scores)

    answer = "\n\n".join([f"Вопрос: {df.iloc[idx, 0]}\t\tОтвет: {df.iloc[idx, 1]}" for idx in top_3_results])

    return OutputData(prediction=answer)



SAIGA_MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
SAIGA_BASE_MODEL_PATH = "TheBloke/Llama-2-7B-fp16"
# BASE_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
SAIGA_DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
SAIGA_DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

class Conversation:
    def __init__(
        self,
        message_template=SAIGA_DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=SAIGA_DEFAULT_SYSTEM_PROMPT,
        start_token_id=1,
        bot_token_id=9225
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()

SAIGA_tokenizer = AutoTokenizer.from_pretrained(SAIGA_MODEL_NAME, use_fast=False)

SAIGA_config = PeftConfig.from_pretrained(SAIGA_MODEL_NAME)
SAIGA_model = AutoModelForCausalLM.from_pretrained(
    SAIGA_BASE_MODEL_PATH,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    # load_in_8bit_fp32_cpu_offload=True,
)
SAIGA_model = PeftModel.from_pretrained(
    SAIGA_model,
    SAIGA_MODEL_NAME,
    torch_dtype=torch.float16
)

SAIGA_model.eval()

generation_config = GenerationConfig.from_pretrained(SAIGA_MODEL_NAME)
print(generation_config)



@app.post("/new_find_similar_saiga", response_model=OutputData)
def predict_new_find_similar_saiga(input_data: InputData):
    client_request = input_data.text

    query_text = f"query: {client_request}"
    query_batch_dict = tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
    query_outputs = model(**query_batch_dict.to(device))
    query_embedding = average_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    scores = (query_embedding @ passage_embeddings.T) * 100

    scores = scores[0].cpu().detach().numpy()
    top_3_results = np.argsort(scores)[-4:][::-1] if len(scores) >= 4 else np.argsort(scores)

    inp = f"У меня есть вопрос: {client_request}. Также у меня есть ответ: {str(df.iloc[top_3_results[0], 1])}. На основе этого ответа коротко ответь на исходный вопрос."
    conversation = Conversation()
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(SAIGA_tokenizer)

    output = generate(SAIGA_model, SAIGA_tokenizer, prompt, generation_config)

    # answer = f"Вопрос: {client_request}\n\nОтвет Cайги: {output}\n\nТоп максимально похожих результатов:\n\n"
    # answer += "\n\n".join([f"Вопрос: {df.iloc[idx, 0]}\t\tОтвет: {df.iloc[idx, 1]}" for idx in top_3_results])

    return OutputData(prediction=output)

@app.post("/test_mock", response_model=OutputData)
def test_mock(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction: dict[str, str] = {
        "prediction": "Тизера нет\n\nВсе еще нет"
    }
    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=str(prediction))




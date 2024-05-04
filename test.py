import requests
from dotenv import load_dotenv
import os
from src.constants import label2text

load_dotenv()

URL = os.getenv("URL")


def test_health_saiga():
    text = ""
    response = requests.post(f"{URL}/saiga", json={"text": text})
    assert response.status_code == 200


def test_health_bert():
    text = ""
    response = requests.post(f"{URL}/classify", json={"text": text})
    assert response.status_code == 200


def test_label():
    text = "Какая единовременная компенсационная выплата при рождении сына?"
    response = requests.post(f"{URL}/classify", json={"text": text})
    assert label2text[response.json()["label"]] == "Детская карта"

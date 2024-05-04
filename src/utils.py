from src.constants import label2text
import psycopg2
import os


def answer_with_label(text, label):
    ans = f"""Тема: {label2text[label]}
    
Ответ: {text}

Пожалуйста, оцените мой ответ:"""
    return ans



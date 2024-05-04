import asyncio
from telebot.async_telebot import AsyncTeleBot
from telebot import types
from dotenv import load_dotenv
import os
import requests
from utils import answer_with_label
import logging
# import psycopg2

load_dotenv()

URL = "https://helping-fwd-ladies-clearance.trycloudflare.com"
PORT = "8080"
token = os.getenv("API_TOKEN")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

logger = logging.getLogger(__name__)
logging.basicConfig(filename='bot.log', encoding='utf-8', level=logging.DEBUG)

bot = AsyncTeleBot(token)

kb = types.InlineKeyboardMarkup([
    [
        types.InlineKeyboardButton(text='1', callback_data='btn_types'),
        types.InlineKeyboardButton(text='2', callback_data='btn_types'),
        types.InlineKeyboardButton(text='3', callback_data='btn_types'),
        types.InlineKeyboardButton(text='4', callback_data='btn_types'),
        types.InlineKeyboardButton(text='5', callback_data='btn_types'),
    ]
])

@bot.message_handler(commands=['start'])
async def send_welcome(message):
    await bot.reply_to(message, """\
Привет, я МФЦ бот.
Я подскажу тебе ответ на любой твой вопрос касающийся работы МФЦ\
""")

@bot.message_handler(func=lambda message: True)
async def message(message):
    logger.info(f"New message: {message.text}")

    res_class = requests.post(f"{URL}/classify", json={"text": message.text})

    if res_class.status_code == 200:

        label = res_class.json()["label"]
        
        if label != 111:

            res_saiga = requests.post(f"{URL}/saiga", json={"text": message.text})

            if res_saiga.status_code == 200:

                text = res_saiga.json()["prediction"]

                await bot.reply_to(message, 
                                   answer_with_label(text, label),
                                   reply_markup=kb)
            else:
                logger.error(res_saiga.text)
                await bot.reply_to(message, "Произошла ошибка при обработке запроса.")
        else:
            await bot.reply_to(message, "Кажется, я не совсем понял вопрос или он не относится к теме МФЦ. Не могли бы вы его уточнить?")

    else:
        logger.error(res_saiga.text)
        await bot.reply_to(message, "Произошла ошибка при обработке запроса.")

# @bot.message_handler(func=lambda message: True)
# def process_rating(func=lambda mesage: )

asyncio.run(bot.polling())
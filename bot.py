import asyncio
from telebot.async_telebot import AsyncTeleBot
from telebot import types
from dotenv import load_dotenv
import os
import requests

load_dotenv()

URL = "https://mandate-ashley-municipal-problems.trycloudflare.com"


token = os.getenv("API_TOKEN")

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

# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
async def send_welcome(message):
    await bot.reply_to(message, """\
Привет, я МФЦ бот.
Я подскажу тебе ответ на любой твой вопрос касающийся работы многофункционального центра\
""")


# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def message(message):
    print(message.text)
    res = requests.post(f"{URL}/new_find_similar_saiga", json={"text": message.text})
    await bot.reply_to(message, res.json()["prediction"], reply_markup=kb)


asyncio.run(bot.polling())
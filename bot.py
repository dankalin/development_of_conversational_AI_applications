import asyncio
from telebot.async_telebot import AsyncTeleBot
from dotenv import load_dotenv
import os
import requests

load_dotenv()

URL = "http://0.0.0.0"
PORT = "8080"


token = os.getenv("API_TOKEN")

bot = AsyncTeleBot(token)

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
    res = requests.post(f"{URL}:{PORT}/new_find_similar_saiga")
    await bot.reply_to(message, res.text)


asyncio.run(bot.pooling())
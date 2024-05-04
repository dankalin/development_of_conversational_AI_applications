import asyncio
from telebot.async_telebot import AsyncTeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv
import os
import requests
import psycopg2

load_dotenv()

URL = "http://0.0.0.0"
PORT = "8080"

token = os.getenv("API_TOKEN")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

bot = AsyncTeleBot(token)

user_states = {}
label2text = {0: '1. Лечение ', 1: '1. Свидетельство 2. Многодетность', 2: 'ВИЧ-инфицированный', 3: 'Выплата в повышенном размере на третьего ребенка', 4: 'Выплата по потере кормильца', 5: 'Герои труда, каваер ордена', 6: 'Детская карта', 7: 'ЕДВ', 8: 'ЕДВ для граждан, достигших возраста 60 и 55 лет (для мужчин и женщин соответственно), которым пенсия не назначена', 9: 'ЕДВ для граждан, которым назначена пенсия (пожизненное содержание) иными органами ', 10: 'ЕДВ для граждан, которым назначена пенсия территориальным органом СФР\n', 11: 'Единовременное пособие при рождении ребенка', 12: 'Единовременное пособие при рождении ребенка в случае прекращения деятельности страхователем либо в случае невозможности его выплаты страхователем', 13: 'Единое пособие', 14: 'Земельный капитал', 15: 'Земельный капитал, земля.', 16: 'Капитальный ремонт, многодетная', 17: 'Капитальный ремонт, многодетная, изменение размера', 18: 'Капитальный ремонт, многодетная, изменение реквизитов', 19: 'Мат. Капитал СПб', 20: 'Мат.помощь', 21: 'Мать-героиня', 22: 'Микроавтобус', 23: 'Паспорт РФ', 24: 'Погасить ипотеку, распоряжение мат. Капиталом', 25: 'Пособие беременной жене военнослужащего по призыву', 26: 'Пособие на ребенка военнослужащего по призыву', 27: 'Пособие от 3 до 7', 28: 'Пособие по беременности и родам в случае невозможности его выплаты страхователем', 29: 'Пособие по беременности и родам, БиР', 30: 'Пособие по уходу', 31: 'Пособие по уходу, в случае невозможности его выплаты страхователем', 32: 'Пособие студенческой семье', 33: 'Пособие школьникам', 34: 'Ребенок инвалид', 35: 'Ребенок с целиакия', 36: 'Ребенок-инвалид', 37: 'Родитель с инвалидностью', 38: 'Родитель с инвалидностью, ребенок-инвалид', 39: 'Рождение ребенка', 40: 'Сад, дет.сад', 41: 'Сад, дет.сад, изменение', 42: 'Сироты, денежное пособие сиротам,  покупка одежды', 43: 'Сироты, единовременное пособие сиротам,  при поступлении', 44: 'Сироты, ежегодное пособие сиротам, литература', 45: 'Сироты, расходы сиротам', 46: 'ТСР ', 47: 'Школа, первый класс', 48: 'Школа, перевод ', 49: 'Школьная форма', 50: 'Экстренная помощь Потери кормильца', 51: 'Экстренная помощь лекарственных препаратов', 52: 'Экстренная помощь,  аваря', 53: 'Экстренная помощь, пожар', 54: 'блокада, ЕДВ, Ленинграде', 55: 'земельный участок', 56: 'путинская выплата', 57: 'сироты, отдых, лечение', 58: 'сироты, проезд сирот', 59: 'социальная помощь, гсп'}

@bot.message_handler(commands=['start'])
async def send_welcome(message):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("Общение с LLM", callback_data="llm"))
    keyboard.add(InlineKeyboardButton("Классификация с BERT", callback_data="bert"))
    await bot.reply_to(message, "Привет! Я МФЦ бот. Выберите режим работы:", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
async def callback_query(call):
    if call.data == "llm":
        await bot.send_message(call.message.chat.id, "Вы выбрали режим общения с LLM. Задайте свой вопрос.")
        user_states[call.message.chat.id] = {"state": "waiting_for_llm_question"}
    elif call.data == "bert":
        await bot.send_message(call.message.chat.id,
                               "Вы выбрали режим классификации с BERT. Введите текст для классификации.")
        user_states[call.message.chat.id] = {"state": "waiting_for_bert_input"}


@bot.message_handler(
    func=lambda message: user_states.get(message.chat.id, {}).get("state") == "waiting_for_llm_question")
async def handle_llm_question(message):
    data = {"text": message.text}
    res = requests.post(f"{URL}:{PORT}/saiga", json=data)

    if res.status_code == 200:
        await bot.reply_to(message, res.json()["prediction"])
    else:
        await bot.reply_to(message, "Произошла ошибка при обработке запроса.")

    await bot.send_message(message.chat.id, "Оцените, пожалуйста, насколько вам помог мой ответ (от 0 до 5):")
    user_states[message.chat.id] = {"state": "waiting_for_llm_rating", "message": message.text}


@bot.message_handler(func=lambda message: user_states.get(message.chat.id, {}).get("state") == "waiting_for_bert_input")
async def handle_bert_input(message):
    data = {"text": message.text}
    res = requests.post(f"{URL}:{PORT}/classify", json=data)

    if res.status_code == 200:
        predicted_class = res.json()["label"]
        predicted_class = label2text[predicted_class]
        await bot.reply_to(message, f"Тег: {predicted_class}")
    else:
        await bot.reply_to(message, "Произошла ошибка при классификации.")

    user_states.pop(message.chat.id, None)


async def process_llm_rating(message):
    try:
        rating = int(message.text)
        if 0 <= rating <= 5:
            user_name = message.from_user.username if message.from_user.username else message.from_user.first_name
            user_message = user_states[message.chat.id]["message"]

            save_rating_to_db(user_name, rating, user_message)

            await bot.reply_to(message, "Спасибо за вашу оценку!")
        else:
            await bot.reply_to(message, "Пожалуйста, введите число от 0 до 5.")
    except ValueError:
        await bot.reply_to(message, "Пожалуйста, введите число от 0 до 5.")
    finally:
        user_states.pop(message.chat.id, None)


@bot.message_handler(func=lambda message: user_states.get(message.chat.id, {}).get("state") == "waiting_for_llm_rating")
async def handle_llm_rating(message):
    try:
        rating = int(message.text)
        if 0 <= rating <= 5:
            user_name = message.from_user.username if message.from_user.username else message.from_user.first_name
            user_message = user_states[message.chat.id]["message"]

            save_rating_to_db(user_name, rating, user_message)

            await bot.reply_to(message, "Спасибо за вашу оценку!")

            # Спрашиваем пользователя, хочет ли он продолжить общение с LLM или перейти в режим классификации
            keyboard = InlineKeyboardMarkup()
            keyboard.add(InlineKeyboardButton("Продолжить общение с LLM", callback_data="continue_llm"))
            keyboard.add(InlineKeyboardButton("Перейти в режим классификации", callback_data="switch_to_bert"))
            await bot.send_message(message.chat.id, "Что вы хотите сделать дальше?", reply_markup=keyboard)

        else:
            await bot.reply_to(message, "Пожалуйста, введите число от 0 до 5.")
    except ValueError:
        await bot.reply_to(message, "Пожалуйста, введите число от 0 до 5.")
    finally:
        user_states.pop(message.chat.id, None)


def save_rating_to_db(user_name, rating, user_message):
    try:
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password
        )

        cur = conn.cursor()

        cur.execute("INSERT INTO statistic (user_name, rating, message) VALUES (%s, %s, %s)",
                    (user_name, rating, user_message))

        conn.commit()
        cur.close()
        conn.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print("Ошибка при сохранении оценки в базу данных:", error)


asyncio.run(bot.polling())

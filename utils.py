from constants import label2text

def answer_with_label(text, label):
    ans = f"""Тема: {label2text[label]}
    
Ответ: {text}"""
    return ans

# def save_rating_to_db(user_name, rating, user_message):
#     try:
#         conn = psycopg2.connect(
#             host=db_host,
#             database=db_name,
#             user=db_user,
#             password=db_password
#         )

#         cur = conn.cursor()

#         cur.execute("INSERT INTO statistic (user_name, rating, message) VALUES (%s, %s, %s)",
#                     (user_name, rating, user_message))

#         conn.commit()
#         cur.close()
#         conn.close()

#     except (Exception, psycopg2.DatabaseError) as error:
#         print("Ошибка при сохранении оценки в базу данных:", error)
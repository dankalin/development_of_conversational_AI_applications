# development_of_conversational_AI_applications

### Чтобы запустить API:
`uvicorn main:app --host 0.0.0.0 --port 8080`

### Чтобы запустить superset:

`docker pull apache/superset`

**Генерируем ключ:**

`openssl rand -base64 42`

**Вставляем ключ вместо `your_secret_key_here`:**

`docker run -d -p 8080:8088 -e "SUPERSET_SECRET_KEY=your_secret_key_here" --name superset apache/superset`

Далее:

`sudo docker exec -it superset superset fab create-admin \
               --username admin \
               --firstname Superset \
               --lastname Admin \
               --email admin@admin.com \
               --password postgres123; \
sudo docker exec -it superset superset db upgrade; \
sudo docker exec -it superset superset load_examples; \
sudo docker exec -it superset superset init;`


### Чтобы поднять postgres:

`sudo docker compose up -d`

### Чтобы запустить telegram бота:

`poetry install`

`python3 main.py`

#### В соседней консоле:

`python3 bot.py`





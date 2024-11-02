from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# Токен вашего бота
TOKEN = "7741273728:AAGBX70IHuRmx8tW5NRgx6kJrsYBOXp4QAU"

# Загрузка и подготовка данных
def load_data():
    # Замените путь на путь к вашему файлу MovieLens
    data = pd.read_csv("ratings.csv")
    movie_features = data.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)
    return movie_features

# Функция для создания и обучения модели
def train_model(movie_features):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(movie_features)
    return model

# Рекомендации для пользователя
def recommend(model, movie_features, user_id, num_recommendations=5):
    distances, indices = model.kneighbors(
        [movie_features.iloc[user_id]], n_neighbors=num_recommendations + 1)
    recommended_movies = [movie_features.index[i]
        for i in indices.flatten()[1:]]
    return recommended_movies

# Обработчик для команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Напишите свой ID пользователя, и я порекомендую фильмы.")

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Ожидается, что пользователь введет числовой ID
        user_id = int(update.message.text)
        recommendations = recommend(model, movie_features, user_id)
        response = f"Рекомендуемые фильмы (ID): {
            ', '.join(map(str, suggestions))}"
            ', '.join(map(str, recommendations))}"
    except ValueError:
        response = "Пожалуйста, введите числовой ID пользователя."
    except IndexError:
        response = "Пользователь с таким ID не найден."
    await update.message.reply_text(response)

# Главная функция для запуска бота
def main():
    application = ApplicationBuilder().token(7741273728:AAGBX70IHuRmx8tW5NRgx6kJrsYBOXp4QAU).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    application.run_polling()

if __name__ == "__main__":
    movie_features = load_data()
    model = train_model(movie_features)
    main()

import base64
import os
import requests
from dotenv import load_dotenv
import telebot

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/generate")

bot = telebot.TeleBot(API_TOKEN)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Welcome to Bot!')


@bot.message_handler(content_types=['photo'])
def call_model(message):

    image_id = message.photo[-1].file_id
    image_info = bot.get_file(image_id) # Fetch file from TG servers
    image = bot.download_file(image_info.file_path)
    bot.reply_to(message, 'Processing image...')

    image_base64 = base64.b64encode(image).decode()
    # Define API endpoint
    response = requests.post(API_URL, json={"image": image_base64})

    bot.reply_to(message, response.json()['generated_text'])


if __name__ == '__main__':
    bot.polling()

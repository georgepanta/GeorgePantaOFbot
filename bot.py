import cv2
import numpy as np
import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import FSInputFile
from fastapi import FastAPI, Request
import uvicorn

# Replace with your actual bot token
BOT_TOKEN = "7950424139:AAFY1tBMt7oRfmOxZq-sQeQRzzfgNctqqqA"
WEBHOOK_URL = "https://georgepantaofbot-aefbdf25db1.herokuapp.com/webhook"

# Create bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# FastAPI app setup
app = FastAPI()

@app.get("/")
async def home():
    return {"status": "Bot is running"}

@app.post("/webhook")
async def webhook(request: Request):
    """ Handle incoming Telegram updates """
    update = await request.json()
    await dp.feed_update(update)
    return {"ok": True}

@dp.message()
async def handle_message(message: types.Message):
    """ Handle messages """
    if message.text == "/start":
        await message.reply("✅ Bot is running! Send me an image.")

async def on_startup():
    """ Set webhook on startup """
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown():
    """ Close bot session on shutdown """
    await bot.session.close()

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    asyncio.run(on_startup())
    uvicorn.run(app, host="0.0.0.0", port=5000)

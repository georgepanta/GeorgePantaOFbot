import cv2
import numpy as np
import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import FSInputFile, Update
from fastapi import FastAPI, Request
import uvicorn

# ✅ Replace with your bot token
BOT_TOKEN = "7950424139:AAFY1tBMt7oRfmOxZq-sQeQRzzfgNctqqqA"
WEBHOOK_URL = "https://georgepantaofbot-aefbdf25db1.herokuapp.com/webhook"

# ✅ Fix: Initialize correctly
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
app = FastAPI()

# ✅ Ensure webhook route exists
@app.post("/webhook")
async def telegram_webhook(request: Request):
    """ Handle incoming Telegram updates """
    try:
        update = Update(**await request.json())
        await dp.feed_update(update)  # ✅ Fix: Use correct update processing
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@dp.message(commands=["start"])
async def start_command(message: types.Message):
    """ Command handler to check if bot is running """
    await message.reply("✅ Bot is online! Send me a photo.")

@dp.message()
async def handle_photo(message: types.Message):
    """ Handles images sent by the user """
    if message.photo:
        await message.reply("✅ Image received! Processing...")
    else:
        await message.reply("⚠️ Please send an image.")

# ✅ Fix Webhook Setup
async def on_startup():
    """ Set webhook on startup """
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown():
    """ Properly close bot session on shutdown """
    await bot.session.close()
    await bot.delete_webhook()

if __name__ == "__main__":
    asyncio.run(on_startup())  # ✅ Fix: Set webhook before running the app
    uvicorn.run(app, host="0.0.0.0", port=5000)

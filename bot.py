import cv2
import numpy as np
import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import FSInputFile, Update
from aiogram.filters import Command
from fastapi import FastAPI, Request
import uvicorn

# Bot Token
BOT_TOKEN = "7592297710:AAEhGzPJnfK5fQhakQYQUzVOaTtwpvYNodc"
WEBHOOK_URL = "https://georgepantaofbot-aefbdf25db1.herokuapp.com/webhook"

# Create bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot=bot)  # ✅ Fix: Properly initialize Dispatcher
app = FastAPI()

# Dictionary to store user images temporarily
user_images = {}

async def process_images(user_id):
    """ Process the model image and chat screenshot """
    if user_id not in user_images or len(user_images[user_id]) < 2:
        return None

    model_image_path, chat_screenshot_path = user_images[user_id]

    # Load images
    model_image = cv2.imread(model_image_path)
    chat_screenshot = cv2.imread(chat_screenshot_path)

    # Convert model image to HSV to detect green screen
    hsv = cv2.cvtColor(model_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of the green area (screen)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest contour (assuming it's the screen)
    screen_contour = max(contours, key=cv2.contourArea)
    screen_pts = cv2.approxPolyDP(screen_contour, 0.02 * cv2.arcLength(screen_contour, True), True)

    if len(screen_pts) != 4:
        return None

    screen_pts = np.float32([point[0] for point in screen_pts])

    # Get the size of the chat screenshot
    h, w, _ = chat_screenshot.shape
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Apply the perspective transformation to fit the screenshot into the green screen area
    matrix = cv2.getPerspectiveTransform(dst_pts, screen_pts)
    chat_screenshot_warped = cv2.warpPerspective(chat_screenshot, matrix, (model_image.shape[1], model_image.shape[0]))

    # Create a mask to replace the green screen with the screenshot
    mask_filled = np.zeros_like(model_image, dtype=np.uint8)
    cv2.fillPoly(mask_filled, [screen_pts.astype(int)], (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask_filled)

    # Remove the green screen and replace it with the screenshot
    model_image_with_screenshot = cv2.bitwise_and(model_image, mask_inv)
    overlay_screen = cv2.bitwise_and(chat_screenshot_warped, mask_filled)
    final_image = cv2.add(model_image_with_screenshot, overlay_screen)

    # Save final image
    output_path = f"output_{user_id}.jpg"
    cv2.imwrite(output_path, final_image)

    return output_path

@dp.message()
async def message_handler(message: types.Message):
    """ Handles all text and photo messages """
    user_id = message.from_user.id

    if message.text:
        if message.text == "/start":
            await message.reply("✅ Bot is online and ready to process images! Send me a photo.")
        else:
            await message.reply("I can only process images. Please send me a photo.")

    elif message.photo:
        photo = message.photo[-1]  # Get the highest resolution version

        # Download the image
        file = await bot.get_file(photo.file_id)
        file_path = file.file_path
        save_path = f"{user_id}_{len(user_images.get(user_id, []))}.jpg"
        await bot.download_file(file_path, save_path)

        # Store the image path
        if user_id not in user_images:
            user_images[user_id] = []
        user_images[user_id].append(save_path)

        # Check if we have both images
        if len(user_images[user_id]) == 2:
            await message.reply("Processing the images... please wait.")
            output_image = await process_images(user_id)

            if output_image and os.path.exists(output_image):
                # Send the processed image back
                await bot.send_photo(user_id, FSInputFile(output_image))
                user_images[user_id] = []  # Clear stored images after processing
            else:
                await message.reply("Something went wrong. Please try again.")

        else:
            await message.reply("Now send me the second image (chat screenshot).")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """ Handle incoming Telegram updates """
    update = Update(**await request.json())
    await dp.feed_update(update)  # ✅ Fix: Properly process the update
    return {"ok": True}

async def on_startup():
    """ Set webhook on startup """
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown():
    """ Properly close the bot session on shutdown """
    await bot.session.close()
    await bot.delete_webhook()

if __name__ == "__main__":
    asyncio.run(on_startup())
    uvicorn.run(app, host="0.0.0.0", port=5000)

import cv2
import numpy as np
import asyncio
import os
import time
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import FSInputFile
from aiogram.filters import Command
from collections import defaultdict
from PIL import Image
from aiogram.utils.exceptions import Throttled

# Configure logging
logging.basicConfig(level=logging.INFO)

# Replace with your Telegram Bot Token
BOT_TOKEN = "7950424139:AAFY1tBMt7oRfmOxZq-sQeQRzzfgNctqqqA"
WEBHOOK_URL = "https://georgepantaofbot-aefbdf25db1.herokuapp.com/webhook"

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Dictionary to store user images temporarily
user_images = {}
user_last_message = defaultdict(int)  # Rate limiting for users

# Resize image if needed (you can customize the size)
def resize_image(image_path, width=800, height=600):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (width, height))
    cv2.imwrite(image_path, img_resized)

# Function to validate images
def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify if the image is corrupted
        return True
    except (IOError, SyntaxError):
        return False

# Image processing function
async def process_images(user_id):
    """ Process the model image and chat screenshot """
    if user_id not in user_images or len(user_images[user_id]) < 2:
        return None

    model_image_path, chat_screenshot_path = user_images[user_id]

    # Resize images before processing
    resize_image(model_image_path)
    resize_image(chat_screenshot_path)

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

# Command: /start
@dp.message(Command("start"))
async def start_command(message: types.Message):
    """ Respond to /start command """
    await message.reply("✅ Bot is online and ready to process images! Send me a photo.")

# Rate limiting: Ensure users don't spam the bot too fast
rate_limit = 3  # seconds
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    current_time = time.time()

    if current_time - user_last_message[user_id] < rate_limit:
        await message.reply("Please wait a moment before sending the next image.")
        return

    user_last_message[user_id] = current_time

    # If message contains no photo, ignore it
    if not message.photo:
        return

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

    # Log the number of images stored for the user
    logging.info(f"Images stored for User ID {user_id}: {len(user_images[user_id])}")

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
        await message.reply("Now wait 3 seconds before sending the screenshot.")

# Auto-restart mechanism to avoid failure if overloaded
async def auto_restart():
    while True:
        # Check if bot has been down for any reason and restart it
        try:
            await bot.get_me()
            logging.info("Bot is online.")
        except Exception as e:
            logging.error(f"Error: {e}. Restarting bot.")
            # Here you could implement any mechanism like an auto-restart after a failure.
            # You can add a restart script on your server or restart dynos on Heroku, etc.
        await asyncio.sleep(300)  # Check every 5 minutes

# Main function to start the bot
async def main():
    task1 = asyncio.create_task(dp.start_polling(bot))
    task2 = asyncio.create_task(auto_restart())
    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    asyncio.run(main())

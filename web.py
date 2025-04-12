from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

@app.post("/generate/")
async def generate_image(model_image: UploadFile = File(...), chat_image: UploadFile = File(...)):
    model_filename = f"{uuid.uuid4()}_model.jpg"
    chat_filename = f"{uuid.uuid4()}_chat.jpg"
    output_filename = f"output_{uuid.uuid4()}.jpg"

    with open(model_filename, "wb") as f:
        f.write(await model_image.read())

    with open(chat_filename, "wb") as f:
        f.write(await chat_image.read())

    # Load images
    model_img = cv2.imread(model_filename)
    chat_img = cv2.imread(chat_filename)

    # Convert model image to HSV to detect green
    hsv = cv2.cvtColor(model_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find green screen contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "Green screen not found."}

    screen = max(contours, key=cv2.contourArea)
    screen_pts = cv2.approxPolyDP(screen, 0.02 * cv2.arcLength(screen, True), True)

    if len(screen_pts) != 4:
        return {"error": "Screen not detected as 4 points."}

    screen_pts = np.float32([pt[0] for pt in screen_pts])
    h, w, _ = chat_img.shape
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(dst_pts, screen_pts)
    warped = cv2.warpPerspective(chat_img, matrix, (model_img.shape[1], model_img.shape[0]))

    mask_filled = np.zeros_like(model_img, dtype=np.uint8)
    cv2.fillPoly(mask_filled, [screen_pts.astype(int)], (255, 255, 255))
    inv_mask = cv2.bitwise_not(mask_filled)

    cleaned_model = cv2.bitwise_and(model_img, inv_mask)
    overlay = cv2.bitwise_and(warped, mask_filled)
    final = cv2.add(cleaned_model, overlay)

    cv2.imwrite(output_filename, final)

    with open(output_filename, "rb") as f:
        content = f.read()

    # Clean temp files
    os.remove(model_filename)
    os.remove(chat_filename)
    os.remove(output_filename)

    return {
        "filename": output_filename,
        "content": content
    }

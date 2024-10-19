import cv2
import numpy as np
import tensorflow as tf
import queue
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load the video and the trained model
video = cv2.VideoCapture('./video/f4k_detection_tracking/gt_122.flv')
model = tf.keras.models.load_model('../model/fish_efficientt.keras')


# Preprocessing function for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Define kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)
backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Queue for frames
frame_queue = queue.Queue()
predictions = queue.Queue()


# Frame reading thread
def read_frames(video):
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_queue.put(frame)


# Start the frame reading thread
threading.Thread(target=read_frames, args=(video,), daemon=True).start()


# Asynchronous prediction function
async def predict_fish(fish_img):
    loop = asyncio.get_event_loop()
    preprocessed_img = preprocess_image(fish_img)
    prediction = await loop.run_in_executor(executor, model.predict, preprocessed_img)
    return np.argmax(prediction)


executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as needed
frame_count = 0

while True:
    if not frame_queue.empty():
        frame = frame_queue.get()

        # Process every other frame
        if frame_count % 2 != 0:
            frame_count += 1
            continue

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))  # Resize the frame for faster processing

        fgmask = backgroundObject.apply(frame)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frameCopy = frame.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) > 250:
                x, y, width, height = cv2.boundingRect(cnt)
                fish_img = frame[y:y + height, x:x + width]  # Crop the detected fish image

                # Perform prediction asynchronously
                class_id = asyncio.run(predict_fish(fish_img))

                # Check if the detected fish is diseased or not
                if class_id == 0:  # Assuming 0 is the class for diseased fish
                    color = (0, 0, 255)  # Red bounding box for diseased fish
                    label = 'Diseased Fish'
                else:
                    color = (0, 255, 0)  # Green bounding box for healthy fish
                    label = 'Healthy Fish'

                cv2.rectangle(frameCopy, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frameCopy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        stacked = np.hstack((frame, frameCopy))
        cv2.imshow('Original Frame and Detected Fishes', cv2.resize(stacked, None, fx=0.8, fy=0.8))

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

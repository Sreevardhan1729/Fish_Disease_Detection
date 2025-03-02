import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vit_b_16
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Setup device as CPU and load the Vision Transformer model (Vit_B_16)
# --------------------------------------------------
device = torch.device("cpu")

# Create the ViT model with num_classes=2 (for two classes: healthy and diseased)
model = vit_b_16(pretrained=False, num_classes=2)

# Load the saved state dict and modify key names to match the model
state_dict = torch.load('../fish_classification/models/ViT.pth', map_location='cpu')
new_state_dict = {}
for k, v in state_dict.items():
    # Replace keys like "heads.0.weight" with "heads.head.weight"
    new_k = k.replace("heads.0", "heads.head")
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# --------------------------------------------------
# Initialize video capture and background subtractor
# --------------------------------------------------
cap = cv2.VideoCapture('./data_sample/fish.mp4')  # Change to 0 for webcam input
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# --------------------------------------------------
# Preprocessing function for cropped fish images
# --------------------------------------------------
def preprocess(crop_img):
    # ViT models typically expect 224x224 images
    input_size = (224, 224)
    img_resized = cv2.resize(crop_img, input_size)
    # Convert from BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1]
    img_norm = img_rgb / 255.0
    # Apply standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    # Convert to tensor (C x H x W) and add batch dimension
    img_tensor = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# --------------------------------------------------
# Main loop for processing video frames
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)
    # Remove shadows by thresholding the mask (optional)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Find contours from the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # Filter out small regions (adjust threshold as needed)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Crop the region (assumed to be a fish)
        fish_roi = frame[y:y+h, x:x+w]

        # Preprocess the cropped image for the ViT model
        input_tensor = preprocess(fish_roi)

        # Perform model inference
        with torch.no_grad():
            output = model(input_tensor)  # output shape: [1,2]
        # Apply softmax to get probabilities for each class
        probs = torch.softmax(output, dim=1)
        # Let's assume class index 1 corresponds to "diseased"
        not_diseased_prob = probs[0, 1].item()

        if not_diseased_prob >= 0.5:
            # Diseased: draw a red box
            color = (0, 255, 0)  # red in BGR
            label = f"Healthy: {not_diseased_prob:.2f}"
        else:
            # Healthy: draw a green box
            color = (0, 0, 255)  # green in BGR

            label = f"Diseased: {1-not_diseased_prob:.2f}"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # Display the processed frame
    cv2.imshow('Fish Disease Detection', frame)
    # Optionally, display the foreground mask:
    # cv2.imshow('Foreground Mask', fgmask)

    # Exit if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

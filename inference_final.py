# inference_esp32_stream.py
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from PIL import Image

# # === Configuration ===
# ESP32_IP = "192.168.25.86"  # Replace with your ESP32-CAM IP
# STREAM_URL = f"http://{ESP32_IP}:81/stream"

class_names = [
    'safe driving',
    'texting - right',
    'talking on the phone - right',
    'texting - left',
    'talking on the phone - left',
    'operating the radio',
    'drinking',
    'reaching behind',
    'hair and makeup',
]

# Load model
model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Video capture from ESP32-CAM
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open stream: {STREAM_URL}")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream disconnected")
            break

        # Convert to PIL format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Inference
        input_tensor = transform(pil_image).unsqueeze(0)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

        # Display
        cv2.putText(frame, f'Predicted: {label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ESP32-CAM Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

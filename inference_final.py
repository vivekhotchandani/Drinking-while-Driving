import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
import torch.nn as nn
from PIL import Image



# -------------------- SETUP -------------------- #
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

# Load and modify the model
model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- INFERENCE LOOP -------------------- #
cap = cv2.VideoCapture(0)

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Apply transforms
        input_tensor = transform(pil_image).unsqueeze(0)

        # Inference
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

        # Show prediction
        cv2.putText(frame, f'Predicted: {label}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-Time Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
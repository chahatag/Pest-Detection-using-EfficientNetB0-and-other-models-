import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load your trained model
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 12)  # 9 classes
    model.load_state_dict(torch.load("C:/Users/chaha/Desktop/PestDetection_Project/efficientnetb0_pest_classifier_1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define class names in order
class_names = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

# Streamlit UI
st.title("🪲 Pest Detection using EfficientNet (PyTorch)")
st.write("Upload a crop image to detect the type of pest.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_index = torch.argmax(probabilities).item()
        pred_class = class_names[pred_index]
        confidence = probabilities[pred_index].item()

    st.success(f"Pest Detected: **{pred_class}** ({confidence * 100:.2f}% confidence)")

    # Show bar chart of all probabilities
    st.subheader(" Prediction Confidence")
    st.bar_chart(probabilities.numpy())

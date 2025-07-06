import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import gdown

import streamlit as st

# Simple access control
password = st.text_input("Enter access code:", type="password")
if password != "datamatrixai":
    st.warning("🔐 Enter the correct password to access the classifier.")
    st.stop()


# ------------------- Configuration -------------------
MODEL_PATH = "best_vit_model.pth"
DRIVE_FILE_ID = "1V-IUkXzZ0pGqN2LsON585FnL9TUdslWw"  # 🔁 Replace with your real Google Drive file ID
CLASS_NAMES = ['Kutcha House', 'Pucca House']
CONFIDENCE_THRESHOLD = 0.90

# ------------------- Download Model -------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ------------------- Preprocessing -------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Batch dimension

# ------------------- Prediction -------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        max_prob = np.max(probs)
        predicted_idx = np.argmax(probs)
        if max_prob < CONFIDENCE_THRESHOLD:
            return "❌ Cannot detect class. Please upload a valid house photo.", probs
        return CLASS_NAMES[predicted_idx], probs

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="🏠 House Type Classifier", layout="centered")

st.title("🏠 House Type Classifier")
st.caption("Identify whether a house is **Kutcha** or **Pucca** from uploaded images.")

# Sidebar
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This AI model classifies house images into two categories:
    - Kutcha House
    - Pucca House

    ⚠️ Irrelevant images are automatically rejected.
    """)
    st.header("👤 Developer")
    st.markdown("""
    **Name:** Shaswat Patra  
    **Email:** patrarishu@gmail.com
    """)

# Load model
model = load_model()

# Upload
uploaded_files = st.file_uploader("Upload house image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.info(f"📷 Processing: `{uploaded_file.name}`")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Predicting..."):
            input_tensor = preprocess_image(image)
            label, probs = predict(model, input_tensor)

        if "Cannot detect" in label:
            st.warning(label)
        else:
            st.success(f"🏷️ Predicted Class: **{label}**")

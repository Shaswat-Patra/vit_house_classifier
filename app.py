# app.py
# A Streamlit app to classify house images as 'Kutcha' or 'Pucca'
import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import os

# Title and sidebar
st.set_page_config(page_title="Smart House Classifier", layout="centered")
st.title(" Smart House Classifier")
st.sidebar.title(" ℹAbout")
st.sidebar.info(
    "This app classifies uploaded house images into 'Kutcha' or 'Pucca' using a trained Swin Transformer model."
)
st.sidebar.title(" Developer Info")
st.sidebar.markdown("**Name:** Shaswat Patra  \n**Email:** xyz.com")


# Load the trained model
@st.cache_resource
def load_model():
    checkpoint_path = "/content/drive/MyDrive/client deliverables/Aashdit/Efficient_dataset(REQ)/vit_models/best_vit_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = timm.create_model(
        "swin_base_patch4_window7_224", pretrained=False, num_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


model, class_names = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform and predict
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)

    if st.button("🔍 Predict"):
        with st.spinner("Classifying..."):
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction = class_names[predicted.item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][
                predicted.item()
            ].item()

            # Confidence threshold (e.g. 80%)
            if confidence < 0.80:
                st.warning(
                    " This image cannot be confidently classified. Please upload a clearer image."
                )
            else:
                st.success(
                    f" Predicted Class: **{prediction}** ({confidence*100:.2f}% confidence)"
                )

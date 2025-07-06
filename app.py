import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import gdown
import os

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

    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

# ------------------- Predict Function -------------------
def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], conf.item()

# ------------------- Streamlit UI -------------------
def main():
    st.set_page_config(page_title="🏠 House Classifier", layout="wide")
    st.title("🏠 House Type Classifier (Kutcha vs Pucca)")
    st.markdown("Upload an image of a house to classify it as a **Kutcha** or **Pucca** house.")
    
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["🏠 Home", "ℹ️ About", "👨‍💻 Developer Info"])

    if app_mode == "🏠 Home":
        uploaded_file = st.file_uploader("Upload image(s) of house", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.divider()
                st.info(f"📷 Processing: `{uploaded_file.name}`")
                
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.success("✅ Image uploaded successfully!")

            if st.button("🔍 Classify Image"):
                model = load_model()
                with st.spinner("⏳ Analyzing image..."):
                    predicted_class, confidence = predict_image(model, image)
                
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ This image doesn't seem to belong to a Kutcha or Pucca house.\nPlease upload a relevant image.")
                else:
                    st.success(f"✅ **Predicted Class:** {predicted_class}")
                    st.info(f"🧠 **Model Confidence:** {confidence*100:.2f}%")

    elif app_mode == "ℹ️ About":
        st.header("About this App")
        st.markdown("""
        This interactive web app uses a pretrained Deep learning model to classify images of houses into:
        - **Kutcha House**
        - **Pucca House**
        
        The model was trained on a custom dataset for client presentation.  
        Please upload only house-level photos for accurate results.
        """)

    elif app_mode == "👨‍💻 Developer Info":
        st.header("Developer Info")
        st.markdown("""
        - 👨‍💻 **Name:** Shaswat Patra  
        - 📧 **Email:** patrarishu@gmail.com  
        - 🛠️ Built with [Streamlit](https://streamlit.io) and [PyTorch](https://pytorch.org)  
        """)

if __name__ == "__main__":
    main()

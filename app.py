import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2
from fpdf import FPDF
from src.model import MyModel, load_model
from src.utils import predict, generate_gradcam
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join("models", "model_38")
model = load_model(model_path, device)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Labels and explanations
label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}

explanation_dict = {
    0: "No Tumor Detected: The MRI scan appears normal with no visible mass or abnormality.\n\nNext Step: Continue with regular health checkups.\nSpecialist: General Physician or Neurologist.",
    1: "Pituitary Tumor Detected: A mass is observed near the pituitary gland, possibly affecting hormonal functions.\n\nSpecialist: Endocrinologist & Neurosurgeon.\nHospitals: Jaslok Hospital (Mumbai), AIIMS (Delhi).\nTreatment: Hormone tests, MRI with contrast, surgery if needed.",
    2: "Glioma Tumor Detected: Gliomas are aggressive brain tumors originating from glial cells.\n\nSpecialist: Neuro-oncologist & Neurosurgeon.\nHospitals: Fortis (Mumbai), Tata Memorial.\nTreatment: Surgery, chemotherapy, radiation.",
    3: "Meningioma Detected: Usually benign tumor arising from the meninges with clear borders.\n\nSpecialist: Neurosurgeon.\nHospitals: Fortis (Delhi), NIMHANS (Bangalore).\nTreatment: Observation or surgical removal if symptomatic.",
    4: "Other Abnormalities: MRI reveals patterns not matching known tumor types.\n\nSpecialist: Neurologist or Neuro-radiologist.\nHospitals: PGIMER (Chandigarh), Apollo Neuro.\nFurther Investigation: Contrast MRI, PET, biopsy."
}

symptom_suggestions = {
    0: "No immediate symptoms. Maintain regular health checkups.",
    1: "Watch for headaches, vision issues, hormonal imbalances.",
    2: "Symptoms may include seizures, cognitive changes, nausea.",
    3: "May cause headaches, vision problems, or motor issues.",
    4: "Symptoms vary. Further tests needed for confirmation."
}

def preprocess_image(image):
    return transform(image).unsqueeze(0)

@st.cache_data
def load_sample_images(sample_images_dir):
    sample_images = []
    for img_name in os.listdir(sample_images_dir):
        path = os.path.join(sample_images_dir, img_name)
        image = Image.open(path).convert("RGB").resize((150, 150))
        sample_images.append((img_name, image))
    return sample_images

def show_gradcam_overlay(image_pil, gradcam_mask):
    img_np = np.array(image_pil.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_mask), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    overlay = heatmap * 0.4 + img_np
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

def extract_feature_maps(model, image_tensor):
    feature_maps = []
    hooks = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu())

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = model(image_tensor.to(device))
    for h in hooks:
        h.remove()
    return feature_maps

def plot_feature_maps(fmaps):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))  # 8 channels in 2x4 grid
    axs = axs.flatten()

    for i in range(8):
        fmap = fmaps[0][0][i]  # Get i-th channel from first feature map
        fmap = fmap.numpy()
        axs[i].imshow(fmap, cmap="viridis")
        axs[i].axis("off")
        axs[i].set_title(f"Feature {i+1}")

    st.pyplot(fig)


# UI Layout
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI to detect tumor type, view model attention using Grad-CAM, view feature maps, and generate a full medical report.")
st.markdown("---")

# Sample Images
st.subheader("📂 Sample MRI Images")
sample_images_dir = "sample"
sample_images = load_sample_images(sample_images_dir)
descriptions = [
    "**Sample 1: Front View (Coronal)** - Divides brain front-back. Good for pituitary evaluation.",
    "**Sample 2: Top View (Axial)** - Top-down cross-section. Good for hemispheric tumors.",
    "**Sample 3: Side View (Sagittal)** - Side-to-side view. Shows brainstem and cerebellum."
]
cols = st.columns(3)
for i, (file, image) in enumerate(sample_images):
    with cols[i]:
        st.image(image, use_container_width=True)
        st.markdown(descriptions[i], unsafe_allow_html=True)

# Upload Section
st.markdown("---")
st.subheader("📤 Upload Your MRI Image")
name = st.text_input("Enter Patient Name")
age = st.text_input("Enter Patient Age")
uploaded_file = st.file_uploader("Upload a JPG image", type="jpg")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=250)
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() * 100

    label = label_dict[predicted_class]
    explanation = explanation_dict[predicted_class]
    symptoms = symptom_suggestions[predicted_class]

    st.markdown("---")
    st.markdown(f"<h2 style='color: #4CAF50;'>✅ Prediction: {label} ({confidence:.2f}% confidence)</h2>", unsafe_allow_html=True)
    st.markdown("### 🧾 Why this classification?")
    st.markdown(f"<div style='background-color: #4169E1; padding: 12px; border-radius: 8px;'>{explanation}</div>", unsafe_allow_html=True)
    st.markdown("### 🩺 Suggested Symptom Check:")
    st.info(symptoms)

    # Grad-CAM
    if st.button("🔥 Show Grad-CAM Heatmap"):
        gradcam_mask = generate_gradcam(model, tensor, predicted_class, device)
        overlay_np = show_gradcam_overlay(image, gradcam_mask)
        gradcam_image = Image.fromarray(overlay_np)

        st.markdown("### 📊 Grad-CAM Visualization")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="🧠 Original MRI", use_column_width=True)
        with col2:
            st.image(gradcam_image, caption="🔥 Grad-CAM", use_column_width=True)

        st.session_state["gradcam"] = gradcam_image

    # Feature maps
    if st.button("🔬 Show CNN Feature Maps"):
        fmap_list = extract_feature_maps(model, tensor)
        plot_feature_maps(fmap_list)

    # PDF Export
    if "gradcam" in st.session_state and st.button("📄 Export PDF Report"):
        original_path = "temp_mri.jpg"
        gradcam_path = "temp_gradcam.jpg"
        image.resize((224, 224)).save(original_path)
        st.session_state["gradcam"].save(gradcam_path)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Brain Tumor MRI Diagnosis Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {label} ({confidence:.2f}% confidence)", ln=True)
        pdf.ln(5)
        explanation_clean = explanation.encode("ascii", "ignore").decode()
        pdf.multi_cell(0, 10, explanation_clean)
        pdf.ln(3)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Suggested Symptom Check:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, symptoms)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Original MRI:", ln=True)
        pdf.image(original_path, w=90)
        pdf.ln(2)
        pdf.cell(200, 10, txt="Grad-CAM Heatmap:", ln=True)
        pdf.image(gradcam_path, w=90)

        pdf_buffer = BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin-1')  # Encode the string output
        pdf_buffer.write(pdf_output)
        pdf_buffer.seek(0)

        st.success("✅ PDF report is ready!")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"Report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Find Neurologists Near Me"):
    st.markdown("[📍 Click here to search on Google Maps](https://www.google.com/maps/search/neurologist+near+me)")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px; color: gray;'>© 2025 | Created for Educational Purposes | Shreeyash Pawar</p>", unsafe_allow_html=True)

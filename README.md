# 🧠 Brain Tumor Classification with Grad-CAM Visualization

This project aims to classify brain tumors from MRI images using a Convolutional Neural Network (CNN) and visualize model attention with Grad-CAM. It provides an intuitive web interface for clinicians and users to understand both predictions and the underlying model focus areas.

---

## 🚀 Features

- **MRI Image Classification** for detecting brain tumors.
- **Deep Learning Model** using CNN trained on a labeled dataset.
- **Grad-CAM Heatmaps** for model interpretability.
- **User-Friendly Streamlit Interface** for image upload and prediction.
- **Performance Metrics** including accuracy, precision, recall, and F1-score.

---

## 🛠 Tech Stack

| Component         | Technology Used          |
|------------------|--------------------------|
| Interface         | Streamlit                |
| Deep Learning     | TensorFlow, Keras        |
| Visualization     | Grad-CAM, Matplotlib     |
| Image Processing  | OpenCV, NumPy            |
| Dataset Handling  | Pandas, Scikit-learn     |

---

## 🧪 Input
- Brain MRI images (.jpg/.png)
- Classified into: **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**

---

## 📁 Project Structure
```
├── model/
│   └── brain_tumor_model.h5
├── cam/
│   └── grad_cam_utils.py
├── app.py
├── static/
│   └── heatmap_output.png
├── NOTEBOOK.PY
│   └── MODEL AND ANALYSIS
├── templates/
│   └── style.css
```

---

## ⚙️ Setup Instructions
1. Clone the repository.
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 📌 Author
**Shreeyash Pawar**  
Department of Electronics & Electrical Engineering, MIT-WPU  
Email: shreeyashpawar0903@gmail.com

---

## 🏁 Future Enhancements
- Integrate patient history for better accuracy.
- Add real-time segmentation of tumor region.
- Enable predictions from DICOM image series.

---


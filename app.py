import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    model_path = 'weights/yolo-crowd.pt'
    if not Path(model_path).exists():
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    try:
        device = select_device('cpu')
        model = attempt_load(model_path, map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def run_detection(image, model, device, img_size=416, conf_thres=0.25, iou_thres=0.45):
    try:
        # Ensure image is in RGB
        img0 = np.array(image.convert('RGB'))
        
        # Preprocess
        img = letterbox(img0, new_shape=img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        # Draw boxes
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred:
                label = f'person {conf:.2f}'
                xyxy = [int(x.item()) for x in xyxy]
                cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                cv2.putText(img0, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return img0, len(pred) if pred is not None else 0
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None, 0

st.title('YOLO-CROWD DETECTION')
st.write('Upload an image to detect number of people.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write('Running detection...')
        
        model, device = load_model()
        if model is not None and device is not None:
            result_img, count = run_detection(image, model, device)
            if result_img is not None:
                st.image(result_img, caption=f'Detected {count} people', use_container_width=True)
                st.success(f'Detection complete: {count} people found.')
            else:
                st.error("Detection failed. Please try again with a different image.")
        else:
            st.error("Failed to load the model. Please check if the model file exists and is accessible.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
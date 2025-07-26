import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def compute_density_map(img, results, num_rows=3, num_cols=3):
    h, w = img.shape[:2]
    cell_h, cell_w = h // num_rows, w // num_cols

    density_map = np.zeros((num_rows, num_cols), dtype=int)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:
            xc = (box[0] + box[2]) / 2
            yc = (box[1] + box[3]) / 2
            row = min(int(yc // cell_h), num_rows - 1)
            col = min(int(xc // cell_w), num_cols - 1)
            density_map[row, col] += 1

    return density_map

def draw_density_overlay(img, density_map):
    num_rows, num_cols = density_map.shape
    cell_h, cell_w = img.shape[0] // num_rows, img.shape[1] // num_cols

    for r in range(num_rows):
        for c in range(num_cols):
            x0, y0 = c * cell_w, r * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)
            cv2.putText(img, str(density_map[r, c]), (x0 + 10, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img

def app():
    st.title("YOLO Crowd Detection + Density Map")

    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = np.array(Image.open(img_file).convert("RGB"))
        results = model(img)

        annotated_img = np.array(results[0].plot())
        density_map = compute_density_map(annotated_img, results)
        output_img = draw_density_overlay(annotated_img.copy(), density_map)

        st.image(output_img, caption="Detected persons + density overlay", use_column_width=True)
        total = int((results[0].boxes.cls.cpu().numpy() == 0).sum())
        st.write(f"🧑 Total Persons Detected: **{total}**")
        st.write("📊 Density per region (rows × cols):")
        st.table(density_map)

if __name__ == "__main__":
    app()

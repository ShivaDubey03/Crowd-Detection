import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def load_image_with_correct_orientation(file):
    image = Image.open(file)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())
        orientation_value = exif.get(orientation)
        if orientation_value == 3:
            image = image.rotate(180, expand=True)
        elif orientation_value == 6:
            image = image.rotate(270, expand=True)
        elif orientation_value == 8:
            image = image.rotate(90, expand=True)
    except Exception:
        pass
    return np.array(image.convert("RGB"))

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
    st.title("Crowd Detection & Density Map")

    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = load_image_with_correct_orientation(img_file)
        results = model(img)

        # Get annotated YOLO image
        annotated_img = np.array(results[0].plot())

        # Count people
        total = int((results[0].boxes.cls.cpu().numpy() == 0).sum())

        # Compute density grid
        density_map = compute_density_map(img, results)
        final_img = draw_density_overlay(annotated_img.copy(), density_map)

        # Draw total people green box on image
        cv2.rectangle(final_img, (10, 10), (270, 60), (0, 255, 0), -1)
        cv2.putText(final_img, f'Total Persons: {total}', (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Display image with overlays
        st.image(final_img, caption="Detections + Region Density", use_container_width=True)
        st.success(f"🧍 Total Persons Detected: {total}")
        st.write("📊 Region-wise person count:")
        st.table(density_map)

# ...existing code...

if __name__ == "__main__":
    app()
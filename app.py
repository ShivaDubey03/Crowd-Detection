import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

def compute_density_map(img, results, num_rows=3, num_cols=3):
    h, w = img.shape[:2]
    cell_h, cell_w = h // num_rows, w // num_cols
    density_map = np.zeros((num_rows, num_cols), dtype=int)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:  # class 0 = person
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

def run_webcam_detection():
    st.subheader("📷 Live Camera Stream")
    cam_placeholder = st.empty()
    counter_placeholder = st.empty()
    stop_btn = st.button("⏹ Stop Camera")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
        return

    while not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to reduce height
        scale = 1
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        results = model(frame)
        boxes = results[0].boxes
        people_indices = (boxes.cls == 0)  # Only keep class 0 = person
        boxes.data = boxes.data[people_indices]

        annotated = results[0].plot()
        count = int(len(boxes.data))

        # Draw count in top-left corner
        cv2.putText(annotated, f"People: {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 1, cv2.LINE_AA)


        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        cam_placeholder.image(annotated, channels="RGB", use_container_width=True)
        counter_placeholder.markdown(f"🧍 Persons Detected: **{count}**")
        time.sleep(0.1)

    cap.release()

def main():
    st.title("SmartCrowd")
    st.caption("Real-Time Person Detection and Density Mapping")

    option = st.radio("Select Mode", ["Upload Image", "Live Webcam"], horizontal=True)

    if option == "Upload Image":
        st.subheader("📸 Upload Image")
        img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if img_file:
            img = np.array(Image.open(img_file).convert("RGB"))
            results = model(img)

            boxes = results[0].boxes
            person_indices = (boxes.cls == 0)
            boxes.data = boxes.data[person_indices]

            annotated_img = np.array(results[0].plot())
            density_map = compute_density_map(annotated_img, results)
            output_img = draw_density_overlay(annotated_img.copy(), density_map)

            st.image(output_img, caption="Detected Persons + Density Overlay", use_container_width=True)
            total = int(len(boxes.data))
            st.success(f"🧑 Total Persons Detected: {total}")
            st.write("📊 Density (Rows × Columns):")
            st.table(density_map)

    elif option == "Live Webcam":
        run_webcam_detection()

if __name__ == "__main__":
    main()

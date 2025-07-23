
# YOLO-CROWD
YOLO-CROWD is a lightweight crowd counting and face detection model that is based on Yolov5s and can run on edge devices, as well as fixing the problems of face occlusion, varying face scales, and other challenges of crowd counting


## Description
Deep learning-based algorithms for face and crowd identification have advanced significantly. These algorithms can be broadly categorized into two groups: one-stage detectors like YOLO and two-stage detectors like Faster R-CNN. One-stage detectors have been widely employed in many applications due to the better balance between accuracy and speed, but as we are all aware, YOLO algorithms are significantly impacted by occlusion in crowd scenarios. In our project, we propose a real-time crowd counter and face detector called **YOLO-CROWD**, which has an inference speed of **10.1 ms** and contains 461 layers and 18388982 parameters. It is based on the one-stage detector YOLOv5. In order to improve the receptive field of small faces, we use a Receptive Field Enhancement module termed RFE. We then use NWD Loss to compensate for the sensitivity of IoU to the position deviation of small objects. We also employ Repulsion Loss to address face occlusion and utilize an attention module called
SEAM.


### Dataset

Download our Dataset [crowd-counting-dataset-w3o7w](https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w), while exporting the dataset select **YOLO v5 PyTorch** Format.

![our-dataset](https://github.com/zaki1003/YOLO-CROWD/assets/65148928/7c574121-7eb5-450c-a61d-d259643d22fb)

## Demo

[streamlit-app-2025-07-24-02-07-48.webm](https://github.com/user-attachments/assets/087f8508-f44c-4522-b5bc-42db2a3194fe)





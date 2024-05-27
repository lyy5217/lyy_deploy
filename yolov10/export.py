from ultralytics import RTDETR, YOLO
"""Test exporting the YOLO model to ONNX format."""
f = YOLO("yolov10s.pt").export(format="onnx", dynamic=True)
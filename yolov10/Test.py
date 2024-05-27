import cv2 as cv
import numpy as np
from openvino.runtime import Core


def load_classes():
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


def format_yolov10(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    return result

# load model
labels = load_classes()
ie = Core()
for device in ie.available_devices:
    print(device)
model = ie.read_model(model="yolov10s.onnx")
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

frame = cv.imread("dog.png")
image = format_yolov10(frame)

h, w, c = image.shape
x_factor = w / 640.0
y_factor = h / 640.0

# 检测 2/255.0, NCHW = 1x3x640x640
blob = cv.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)

# 设置网络输入
cvOut = compiled_model([blob])[output_layer]
# [left,top, right, bottom, score, classId]
print(cvOut.shape)
for row in cvOut[0, :, :]:
    score = float(row[4])
    objIndex = int(row[5])
    if score > 0.5:
        left, top, right, bottom = row[0].item(), row[1].item(), row[2].item(), row[3].item()

        left = int(left * x_factor)
        top = int(top * y_factor)
        right = int(right * x_factor)
        bottom = int(bottom * y_factor)
        # 绘制
        cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
        cv.putText(frame, "score:%.2f, %s"%(score, labels[objIndex]),
                (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8);

cv.imshow('YOLOv10 Object Detection', frame)
cv.imwrite("result.png", frame)
cv.waitKey(0)
cv.destroyAllWindows()

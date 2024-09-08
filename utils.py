import cv2
from datetime import datetime
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.config.fd_config import define_img_size
import sys

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming</title>
</head>
<body>
<div style = "text-align: center">
<h1>Picamera2 MJPEG Streaming</h1>
<img src="stream.mjpg" style = "width: 90%"/>
</div>
</body>
</html>
"""


class BooleanObject:
    def __init__(self, value):
        self.value = bool(value)  # 确保布尔值被正确封装
 
    def get_value(self):
        return self.value
 
    def set_value(self, value):
        self.value = bool(value)

def default_handler(frame):
    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, str(time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def person_detect(orig_image):
    define_img_size(320)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
    label_path = "./models/voc-model-labels.txt"
    test_device = "cpu"
    net_type = "RFB"
    candidate_size = 1500
    threshold = 0.5

    class_names = [name.strip() for name in open(label_path).readlines()]
    if net_type == 'slim':
        model_path = "models/pretrained/version-slim-320.pth"
        # model_path = "models/pretrained/version-slim-640.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    elif net_type == 'RFB':
        model_path = "models/train-version-RFB/RFB-Epoch-299-Loss-2.968528504555042.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)

    sum = 0
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    probs = probs.numpy()
    sum += boxes.size(0)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        b = probs[i]
        label = f"{probs[i]:.2f}"
    cv2.putText(image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('video',image)
    print(f"Found {len(probs)} faces.")


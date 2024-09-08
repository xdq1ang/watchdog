"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys
import numpy
import cv2
from picamera2 import Picamera2
import time
from vision.ssd.config.fd_config import define_img_size
from utils import person_detect

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "./models/voc-model-labels.txt"
test_device = args.test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
elif args.net_type == 'RFB':
    model_path = "models/train-version-RFB/RFB-Epoch-299-Loss-2.968528504555042.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

camera_width = 640
camera_height = 480
camera = Picamera2()
camera.video_configuration.controls.FrameRate = 60.0
camera_config = camera.create_preview_configuration(
    main={
        'size': (camera_width, camera_height)
    }
)
camera.configure(camera_config)
camera.start()

sum = 0
while True:
    # time.sleep(1)
    orig_image = camera.capture_array()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    probs = probs.numpy()
    # sum += boxes.size(0)
    # for i in range(boxes.size(0)):
    #     box = boxes[i, :]
    #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    #     b = probs[i]
    #     label = f"{probs[i]:.2f}"
    # cv2.putText(image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.imshow('video',image)
    if len(probs) != 0:
        print(f"detected person !")

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
camera.close()
print(sum)

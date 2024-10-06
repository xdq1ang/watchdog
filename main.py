from collections import Counter
import atexit
import os
from multiprocessing import Process, Queue
from mjpg import StreamingServer, StreamingHandler
from cfg import streaming_config
import cv2
import numpy as np
from datetime import datetime
from scipy.signal import convolve
from picamera2 import Picamera2
import logging
import resource

logging.basicConfig(level=logging.INFO)

def limit_memory(max_memory_mb):
    max_memory_byte = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_RSS, (max_memory_byte, max_memory_byte))

@atexit.register
def when_exit():
    worker0.close()
    worker1.close()

def schmidt_rectification(data, low_threshold, high_threshold):
    output = np.zeros_like(data)
    state = 0
    slices = []
    start, end = -1, -1
    for i in range(len(data)):
        if state == 0:
            if data[i] > high_threshold:
                state = 1
                output[i] = 1
                start = i
            else:
                output[i] = 0
        elif state == 1:
            if data[i] < low_threshold:
                state = 0
                output[i] = 0
                end = i
                slices.append((start, end))
            else:
                output[i] = 1
    return output, slices

def dump_video(frames: list[np.ndarray], times, filename: str):
    frame = frames[0]
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, 10.0, (frame_width, frame_height), isColor=True)

    for frame, time in zip(frames, times):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, str(time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()

def cache_handler(q: Queue, chunk_size: int, save_root: str):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg_model = cv2.createBackgroundSubtractorMOG2()
    hanning = np.hanning(11)

    ratios = []

    frames = []
    times = []
    iterations = 0
    gap_ratio = 3

    while True:
        frame: np.ndarray = q.get()
        now = datetime.now()
        camera_size = frame.shape[0] * frame.shape[1]

        frames.append(frame)
        times.append(now.strftime('%Y年%m月%d日%H:%M:%S'))

        if iterations % gap_ratio == 0:
            bg_mask = bg_model.apply(frame)
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
            _, bg_mask = cv2.threshold(bg_mask, 100, 255, cv2.THRESH_BINARY)
            ratio = Counter(bg_mask.flatten()).get(255, 0) / camera_size
            ratios.append(ratio)
            if iterations > 99999999:
                iterations = 0

        iterations += 1

        # 检测到运动时保存视频
        if len(ratios) >= chunk_size:
            conv_signal = convolve(ratios, hanning, mode='same')
            sch_signal, slices = schmidt_rectification(conv_signal, 0.2, 0.45)
            for start, end in slices:
                real_start = start * gap_ratio
                real_end = end * gap_ratio
                dump_frames = frames[real_start: real_end + 1]
                dump_times = times[real_start: real_end + 1]
                filename = f'{times[real_start]}~{times[real_end]}.avi'
                save_path = os.path.join(save_root, filename)
                dump_video(dump_frames, dump_times, save_path)
                # print(f'保存视频到 {save_path}')
                logging.info(f'保存视频到 {save_path}')
            frames.clear()
            times.clear()
            ratios.clear()

def start_streaming(streaming_status):
    try:
        address = ('', 7123)
        StreamingHandler.streaming_status = streaming_status
        server = StreamingServer(address, StreamingHandler)
        logging.info('通过 http://192.168.6.149:7123/index.html 来访问推流')
        server.serve_forever()
    finally:
        logging.info('结束推流')
        camera.stop_recording()

if __name__ == '__main__':
    limit_memory(512)

    save_root = 'videos'
    os.makedirs(save_root, exist_ok=True)

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

    chunk_size = 50
    queue_detect = Queue(maxsize=chunk_size)
    queue_streaming = Queue(maxsize=chunk_size)

    worker0 = Process(target=cache_handler, args=(queue_detect, chunk_size, save_root))
    worker0.start()

    streaming_config['queue'] = queue_streaming
    # 父子进程同步串流状态, 允许10个串流同时进行
    streaming_status = Queue(10)
    worker1 = Process(target=start_streaming, args=(streaming_status, ))
    worker1.start()


    
    while True:
    
        frame = camera.capture_array()
        queue_detect.put(frame)

        if streaming_status.qsize() != 0:
            queue_streaming.put(frame)

        # print("queue_detect: ", queue_detect.qsize())
        # print("queue_streaming: ", queue_streaming.qsize())
import logging
import socketserver
from http import server
import numpy as np
import cv2
from cfg import streaming_config



class StreamingHandler(server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        html = streaming_config['html']
        frame_handler = streaming_config['frame_handler']

        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                StreamingHandler.streaming_status.put("someone request the url, please start streaming !")
                queue = streaming_config['queue']
                while True:
                    frame: np.ndarray = queue.get()
                    frame = frame_handler(frame)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    _, frame = cv2.imencode('.jpg', frame)
                    frame_bytes = frame.tobytes()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                StreamingHandler.streaming_status.get()
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

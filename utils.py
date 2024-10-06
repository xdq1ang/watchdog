
PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
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
    return frame


import cv2
import numpy as np
import http.server
import socketserver
import threading
import os

img_path = r"E:\photos\selected\selected\frame_000042.png"
img = cv2.imread(img_path)

LANE_CONFIG = {
    "left_road": [(42, 660), (38, 392), (407, 274), (499, 646)],
    "bottom_road": [(1073, 327), (1280, 641), (1280, 720), (678, 720)],
    "right_road": [(1018, 45), (1108, 119), (1050, 357), (709, 161)],
    "top_road": [(362, 25), (447, 10), (807, 176), (432, 246)],
    "top_road_NEW": [(401, 146), (642, 100), (807, 176), (432, 246)]
}

colors = {
    "left_road": (255, 0, 0),
    "bottom_road": (0, 255, 0),
    "right_road": (0, 0, 255),
    "top_road": (0, 255, 255),
    "top_road_NEW": (255, 0, 255) # Magenta
}

for lane, pts in LANE_CONFIG.items():
    pts_arr = np.array(pts, np.int32)
    pts_arr = pts_arr.reshape((-1, 1, 2))
    cv2.polylines(img, [pts_arr], isClosed=True, color=colors[lane], thickness=3)

cv2.imwrite("test_polygons.jpg", img)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        super().end_headers()

PORT = 8000
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()

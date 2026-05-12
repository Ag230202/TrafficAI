import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r"E:\photos\selected\selected\frame_000042.png"
img = cv2.imread(img_path)

LANE_CONFIG = {
    "left_road": [(42, 660), (38, 392), (407, 274), (499, 646)],
    "bottom_road": [(1073, 327), (1280, 641), (1280, 720), (678, 720)],
    "right_road": [(1018, 45), (1108, 119), (1050, 357), (709, 161)],
    "top_road": [(362, 25), (447, 10), (807, 176), (432, 246)]
}

colors = {
    "left_road": (255, 0, 0),    # Blue
    "bottom_road": (0, 255, 0),  # Green
    "right_road": (0, 0, 255),   # Red
    "top_road": (0, 255, 255)    # Yellow
}

for lane, pts in LANE_CONFIG.items():
    pts_arr = np.array(pts, np.int32)
    pts_arr = pts_arr.reshape((-1, 1, 2))
    cv2.polylines(img, [pts_arr], isClosed=True, color=colors[lane], thickness=3)
    
    # Put text
    M = cv2.moments(pts_arr)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = pts[0]
    cv2.putText(img, lane, (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[lane], 2)

cv2.imwrite("polygons_debug.jpg", img)
print("Saved to polygons_debug.jpg")

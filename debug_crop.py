import cv2
import numpy as np

img_path = r"E:\photos\selected\selected\frame_000042.png"
img = cv2.imread(img_path)

# Original configs
LANE_CONFIG = {
    "left_road_OLD": [(42, 660), (38, 392), (407, 274), (499, 646)],
    "right_road_OLD": [(1018, 45), (1108, 119), (1050, 357), (709, 161)],
    "bottom_road": [(1073, 327), (1280, 641), (1280, 720), (678, 720)],
    "top_road_NEW": [(401, 146), (642, 100), (807, 176), (432, 246)]
}

# Crop left_road halfway
# Top edge: (38, 392) to (407, 274) -> midpoint (222, 333)
# Bottom edge: (42, 660) to (499, 646) -> midpoint (270, 653)
LANE_CONFIG["left_road_NEW"] = [(270, 653), (222, 333), (407, 274), (499, 646)]

# Crop right_road halfway
# Top edge: (1018, 45) to (709, 161) -> midpoint (863, 103)
# Bottom edge: (1108, 119) to (1050, 357) -> midpoint (1079, 238)
LANE_CONFIG["right_road_NEW"] = [(863, 103), (1079, 238), (1050, 357), (709, 161)]

colors = {
    "left_road_OLD": (255, 0, 0),     # Blue
    "left_road_NEW": (255, 165, 0),   # Orange
    "right_road_OLD": (0, 0, 255),    # Red
    "right_road_NEW": (255, 0, 255),  # Magenta
    "bottom_road": (0, 255, 0),       # Green
    "top_road_NEW": (0, 255, 255)     # Yellow
}

for lane, pts in LANE_CONFIG.items():
    pts_arr = np.array(pts, np.int32)
    pts_arr = pts_arr.reshape((-1, 1, 2))
    thickness = 3 if "NEW" in lane or lane == "bottom_road" else 1
    cv2.polylines(img, [pts_arr], isClosed=True, color=colors[lane], thickness=thickness)

cv2.imwrite("test_polygons2.jpg", img)
print("Saved to test_polygons2.jpg")

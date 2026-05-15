import cv2
import numpy as np

# ---- IMAGE PATH ----
img = cv2.imread(r"D:\Traffic_AI\Traffic_Footage_Sanity\ezgif-frame-006.jpg")

# ---- LANE CONFIG ----
LANE_CONFIG = {
    "left_road": [(8,399),(397,256),(492,720),(0,720)],
    "bottom_road": [(1083,411),(1269,635),(1277,714),(801,717)],
    "right_road": [(1005,81),(1100,134),(1100,400),(1028,304),(784,164)],
    "top_road": [(405,51),(466,40),(723,159),(455,209)]
}

# ---- OPTIONAL GLOBAL SHIFT ----
dx, dy = 0, 0  # change if needed

for lane in LANE_CONFIG:
    LANE_CONFIG[lane] = [(x+dx, y+dy) for (x,y) in LANE_CONFIG[lane]]

# ---- COLORS ----
colors = {
    "left_road": (255,0,0),
    "bottom_road": (0,255,0),
    "right_road": (0,0,255),
    "top_road": (0,255,255)
}

# ---- DRAW ----
output = img.copy()

for name, pts in LANE_CONFIG.items():
    pts_np = np.array(pts, np.int32)

    # transparent fill
    overlay = output.copy()
    cv2.fillPoly(overlay, [pts_np], colors[name])
    output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

    # border
    cv2.polylines(output, [pts_np], True, colors[name], 2)

    # label
    x, y = pts[0]
    cv2.putText(output, name, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)

# ---- SHOW ----
cv2.imshow("Lane Verification", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for getting points manually, use the code below

# import cv2

# points = []
# img = cv2.imread(r"D:\Traffic_AI\Traffic_Footage_Sanity\ezgif-frame-006.jpg")

# def click(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         print(points)

#         # draw point
#         cv2.circle(img, (x,y), 5, (0,255,0), -1)
#         cv2.imshow("img", img)

# cv2.imshow("img", img)
# cv2.setMouseCallback("img", click)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("FINAL POINTS:", points)
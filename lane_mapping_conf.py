# import cv2
# import numpy as np

# # ---- IMAGE PATH ----
# img = cv2.imread(r"E:\photos\selected\selected\frame_000042.png")

# # ---- LANE CONFIG ----
# LANE_CONFIG = {
#     "left_road": [(42, 660), (38, 392), (407, 274), (499, 646)],
#     "bottom_road": [(1073, 327), (1280, 641), (1280, 720), (678, 720)],
#     "right_road": [(1018, 45), (1108, 119), (1050, 357), (709, 161)],
#     "top_road": [(362, 25), (447, 10), (807, 176), (432, 246)]
# }

# # ---- OPTIONAL GLOBAL SHIFT ----
# dx, dy = 0, 0  # change if needed

# for lane in LANE_CONFIG:
#     LANE_CONFIG[lane] = [(x+dx, y+dy) for (x,y) in LANE_CONFIG[lane]]

# # ---- COLORS ----
# colors = {
#     "left_road": (255,0,0),
#     "bottom_road": (0,255,0),
#     "right_road": (0,0,255),
#     "top_road": (0,255,255)
# }

# # ---- DRAW ----
# output = img.copy()

# for name, pts in LANE_CONFIG.items():
#     pts_np = np.array(pts, np.int32)

#     # transparent fill
#     overlay = output.copy()
#     cv2.fillPoly(overlay, [pts_np], colors[name])
#     output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)

#     # border
#     cv2.polylines(output, [pts_np], True, colors[name], 2)

#     # label
#     x, y = pts[0]
#     cv2.putText(output, name, (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)

# # ---- SHOW ----
# cv2.imshow("Lane Verification", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for getting points manually, use the code below

import cv2

points = []
img = cv2.imread(r"E:\photos\selected\selected\frame_000042.png")

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(points)

        # draw point
        cv2.circle(img, (x,y), 5, (0,255,0), -1)
        cv2.imshow("img", img)

cv2.imshow("img", img)
cv2.setMouseCallback("img", click)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("FINAL POINTS:", points)
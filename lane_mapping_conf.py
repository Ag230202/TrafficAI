import cv2
import numpy as np
from lane_mapper import LANE_CONFIG

# ---- IMAGE PATH ----
# Using the footage where you just picked the coordinates
img = cv2.imread(r"D:\Traffic_AI\aa\frame_000001.png")

# ---- COLORS ----
colors = {
    "left_road": (255,0,0),
    "bottom_road": (0,255,0),
    "right_road": (0,0,255),
    "top_road": (0,255,255)
}

# ---- DRAW ----
if img is not None:
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
    cv2.imshow("Final Lane Verification", output)
    print("[INFO] Showing final lanes. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[ERROR] Could not find the image path.")


# --- MANUAL POINT PICKER (Keep for future recalibration) ---
# points = []
# def click(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         print(points)
#         cv2.circle(img, (x,y), 5, (0,255,0), -1)
#         cv2.imshow("img", img)
#
# if img is not None:
#     cv2.imshow("img", img)
#     cv2.setMouseCallback("img", click)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("FINAL POINTS:", points)
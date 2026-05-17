import cv2
import numpy as np
from lane_mapper import LANE_CONFIG

# ---- IMAGE PATH ----
img_path = r"D:\Traffic_AI\aa\frame_000001.png"
img = cv2.imread(img_path)

# ---- COLORS ----
colors = {
    "left_road": (255, 0, 0),        # Blue (OpenCV BGR: Blue is (255,0,0))
    "bottom_road": (0, 255, 0),      # Green
    "right_road": (0, 0, 255),       # Red
    "top_road": (0, 255, 255),       # Yellow
    "intersection_center": (255, 0, 255) # Magenta
}

points = []

def click_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[POINT ADDED] Vertex #{len(points)} plotted at: ({x}, {y})")
        print(f"Current Coordinate List: {points}")
        
        # Redraw screen with new vertices
        redraw()

def redraw():
    global points
    display_img = base_output.copy()
    
    # Draw interactive points
    for idx, pt in enumerate(points):
        # Draw red dot
        cv2.circle(display_img, pt, 6, (0, 0, 255), -1)
        # Label with sequence number
        cv2.putText(display_img, str(idx + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    # Draw lines connecting the vertices
    if len(points) > 1:
        pts_np = np.array(points, np.int32)
        cv2.polylines(display_img, [pts_np], False, (0, 0, 255), 2)
        if len(points) >= 3:
            # Draw preview line from last to first
            cv2.line(display_img, points[-1], points[0], (0, 165, 255), 1, cv2.LINE_AA)
            
    cv2.imshow("Lane Calibration & Point Picker", display_img)

if img is not None:
    # 1. Resize image to standard 1280x720 first so coordinates match the pipeline exactly
    img = cv2.resize(img, (1280, 720))
    base_output = img.copy()

    # 2. Draw existing lane polygons for context (excluding right_road so you have a clean slate)
    for name, pts in LANE_CONFIG.items():
        if name == "right_road":
            continue
        pts_np = np.array(pts, np.int32)

        # Transparent fill
        overlay = base_output.copy()
        cv2.fillPoly(overlay, [pts_np], colors.get(name, (0, 255, 255)))
        base_output = cv2.addWeighted(overlay, 0.3, base_output, 0.7, 0)

        # Border
        cv2.polylines(base_output, [pts_np], True, colors.get(name, (0, 255, 255)), 2)

        # Label
        x, y = pts[0]
        cv2.putText(base_output, name, (x, y - 10 if y > 20 else y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors.get(name, (0, 255, 255)), 2)

    print("\n" + "="*80)
    print(" 🛠️  INTERACTIVE LANE CALIBRATION & POINT PICKER MODE (RIGHT ROAD)")
    print("="*80)
    print(" Instructions:")
    print("  1. LEFT-CLICK on the image window to plot vertices for the 'right_road'.")
    print("  2. Vertices will connect sequentially in red.")
    print("  3. Coordinates print in your terminal in real-time as you click.")
    print("  4. Once you have closed the loop, PRESS ANY KEY on your keyboard to save.")
    print("  5. Copy-paste the final list from your terminal and paste it here!")
    print("="*80 + "\n")

    cv2.namedWindow("Lane Calibration & Point Picker")
    cv2.setMouseCallback("Lane Calibration & Point Picker", click_callback)
    
    # Draw initial state
    redraw()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*80)
    print(" 🎉 CALIBRATION COMPLETED!")
    print(f" Final coordinates for your new polygon:")
    print(f" {points}")
    print("="*80 + "\n")
else:
    print("[ERROR] Could not find the image path.")
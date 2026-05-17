import cv2

def main():
    img_path = r"D:\Traffic_AI\aa\frame_000001.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image.")
        return
    print(f"Original shape of the frame: {img.shape}")

if __name__ == "__main__":
    main()

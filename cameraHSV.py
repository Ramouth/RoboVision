import cv2
import numpy as np

# --- HSV Color Ranges ---
WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 55, 255])
ORANGE_LOWER = np.array([5, 150, 150])
ORANGE_UPPER = np.array([15, 255, 255])

# --- Detection Functions ---
def color_mask_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    mask_orange = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    return cv2.bitwise_or(mask_white, mask_orange)

def detect_circles(mask, dp=1.2, min_dist=15, param1=50, param2=30, min_radius=10, max_radius=50):
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    return np.round(circles[0, :]).astype("int") if circles is not None else []

def draw_detected_balls(frame, circles):
    for (x, y, r) in circles:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return frame

# --- Main Loop ---
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    else:
        print("Webcam successfully opened.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            mask = color_mask_hsv(frame)
            circles = detect_circles(mask)
            frame = draw_detected_balls(frame, circles)

            cv2.imshow('Ping Pong Ball Detector (HSV Only)', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
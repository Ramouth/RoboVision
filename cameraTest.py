import cv2
import numpy as np

# --- HSV Color Ranges ---
# More restrictive white threshold (less likely to pick up gray/white clothing)
WHITE_LOWER = np.array([0, 0, 220])  # Increased V min value
WHITE_UPPER = np.array([180, 30, 255])  # Decreased S max value
# More specific orange threshold
ORANGE_LOWER = np.array([8, 170, 170])  # Narrower range
ORANGE_UPPER = np.array([12, 255, 255])  # Narrower range

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

def detect_position(x, frame_width):
    # Divide the screen into three equal parts
    left_bound = frame_width // 3
    right_bound = 2 * frame_width // 3
    
    if x < left_bound:
        return "Left"
    elif x > right_bound:
        return "Right"
    else:
        return "Forward"

def draw_detected_balls(frame, circles):
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Counter for balls
    ball_count = len(circles)
    
    # Draw screen divisions (for visualization)
    left_bound = width // 3
    right_bound = 2 * width // 3
    cv2.line(frame, (left_bound, 0), (left_bound, height), (255, 0, 0), 1)
    cv2.line(frame, (right_bound, 0), (right_bound, height), (255, 0, 0), 1)
    
    # Position text to display
    position_text = ""
    
    for (x, y, r) in circles:
        # Draw circle and center
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        # Determine position of the ball
        ball_position = detect_position(x, width)
        
        # Label each ball with its position
        cv2.putText(frame, ball_position, (x - 30, y - r - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # For multiple balls, we'll use the closest one to center as the main position
        # But for now, just use the last ball's position
        position_text = ball_position
    
    # Display ball count
    cv2.putText(frame, "Ball Count: " + str(ball_count), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display main position directive
    if position_text:
        cv2.putText(frame, "Direction: " + position_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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
            
            cv2.imshow('Ping Pong Ball Detector', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
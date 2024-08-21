import cv2
import time
from config import DISPLAY_VIDEO
from diagnostics import get_diagnostics
from detector import detect_and_estimate, draw_keypoints, detect_pose

# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect human and estimate distance
    print("Calling detect_and_estimate")
    x_center, y_center, processed_frame, person_confidences = detect_and_estimate(frame)
    print(f"Person confidences: {person_confidences}")

    # Detect pose (limb tracking) for multiple people
    print("Calling detect_pose")
    people_keypoints, people_scores = detect_pose(processed_frame)
    print(f"Keypoints detected: {people_keypoints}")
    print(f"Scores detected: {people_scores}")

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # Draw keypoints and confidence scores on the frame
    print("Calling draw_keypoints")
    processed_frame = draw_keypoints(processed_frame, people_keypoints, people_scores, person_confidences)

    # Get diagnostics
    cpu_usage, ram_usage = get_diagnostics()

    if DISPLAY_VIDEO:
        # Display diagnostics on the frame
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"CPU: {cpu_usage:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"RAM: {ram_usage:.2f} MB", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the video stream
        print("Displaying frame")
        cv2.imshow("Frame", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print relevant values if not displaying video
    if not DISPLAY_VIDEO:
        print(f"X: {x_center}, Y: {y_center}, CPU: {cpu_usage}%, RAM: {ram_usage:.2f} MB")

cap.release()
cv2.destroyAllWindows()

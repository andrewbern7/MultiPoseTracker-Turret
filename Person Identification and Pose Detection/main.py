import cv2
import time
from safeword_listener import SafewordListener
from config import DISPLAY_VIDEO, vosk_model_path, SAFEWORD
from diagnostics import get_diagnostics
from detector import detect_and_estimate, draw_keypoints, detect_pose

# Set up the safeword listener
safeword_listener = SafewordListener(safeword=SAFEWORD, model_path=vosk_model_path)
safeword_listener.start()

# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

while not safeword_listener.is_shutdown_triggered():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect human and estimate distance
    x_center, y_center, processed_frame, person_confidences = detect_and_estimate(frame)

    # Detect pose (limb tracking) for multiple people
    people_keypoints, people_scores = detect_pose(processed_frame)
    processed_frame = draw_keypoints(processed_frame, people_keypoints, people_scores, person_confidences)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # Display FPS on the frame
    cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Get diagnostics
    cpu_usage, ram_usage = get_diagnostics()

    if DISPLAY_VIDEO:
        # Display diagnostics on the frame
        cv2.putText(processed_frame, f"CPU: {cpu_usage:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"RAM: {ram_usage:.2f} MB", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the video stream
        cv2.imshow("Frame", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Reset FPS calculation every 60 frames
    if frame_count == 60:
        frame_count = 0
        start_time = time.time()

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Turret system shut down.")

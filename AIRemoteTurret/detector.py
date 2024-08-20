import cv2
import numpy as np
import tensorflow as tf
from config import FOCAL_LENGTH, OBJECT_WIDTH, DISPLAY_VIDEO

# Load the pre-trained MobileNet SSD model for human detection
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')

# Load the MoveNet MultiPose model for pose estimation
interpreter = tf.lite.Interpreter(model_path='models/movenet_multipose_lightning_float16.tflite')
interpreter.allocate_tensors()

# Get input and output details for MoveNet MultiPose
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def detect_and_estimate(frame):
    """Detect humans in the frame and estimate distance."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    x_center, y_center = 0, 0
    person_confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # 15 is the class ID for a person
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                perceived_width = endX - startX

                # Estimate distance
                distance = (OBJECT_WIDTH * FOCAL_LENGTH) / perceived_width

                # Calculate center of mass (x, y)
                x_center = (startX + endX) // 2
                y_center = (startY + endY) // 2

                if DISPLAY_VIDEO:
                    # Draw bounding box and center of mass
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)  # Draw Center of Mass
                    cv2.putText(frame, f"Distance: {distance:.2f}m", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Store the person detection confidence
                    person_confidences.append((confidence, startX, startY, endX, endY))

    return x_center, y_center, frame, person_confidences


def preprocess_image(frame):
    """Preprocess image to the required size and shape for MoveNet MultiPose."""
    input_shape = input_details[0]['shape']
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0).astype(np.uint8)  # MoveNet expects uint8 input
    return image


def detect_pose(frame):
    """Detect multiple pose landmarks in the frame using MoveNet MultiPose."""
    preprocessed_image = preprocess_image(frame)

    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()

    # Extract the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    people_keypoints = []
    people_scores = []

    for person_data in output_data[0]:
        # The first 51 elements are the keypoints (17 keypoints * 3 values each)
        keypoints = person_data[:51].reshape((17, 3))  # Reshape to (17, 3)
        keypoint_coords = keypoints[:, :2]  # Extract y, x coordinates
        keypoint_scores = keypoints[:, 2]  # Extract the confidence scores

        people_keypoints.append(keypoint_coords)
        people_scores.append(keypoint_scores)

    return people_keypoints, people_scores


def draw_keypoints(frame, people_keypoints, people_scores, person_confidences, threshold=0.5):
    """Draw keypoints for multiple people on the frame using MoveNet MultiPose."""
    height, width, _ = frame.shape

    # Draw keypoints for each detected person
    for keypoints, scores, (confidence, startX, startY, endX, endY) in zip(people_keypoints, people_scores,
                                                                           person_confidences):
        for i, score in enumerate(scores):
            if score > threshold:
                y, x = int(keypoints[i][0] * height), int(keypoints[i][1] * width)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw confidence score on the bounding box
        cv2.putText(frame, f"Conf: {confidence:.2f}", (endX, endY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

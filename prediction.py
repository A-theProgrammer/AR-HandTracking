import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2
import time

# Create a MediaPipe HandLandmarker detector. 
# Requires MediaPipe 0.9.1 and above.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def predict(frame):
    """
    Implement the hand landmark prediction.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    return detection_result

def draw_landmarks_on_image(image, detection_result):
    """
    A helper function to draw the detected 2D landmarks on an image 
    """
    if not detection_result:
        return image 
    
    hand_landmarks_list = detection_result.hand_landmarks
    # Loop through the detected hands and draw directly on the image
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return image

def get_camera_matrix(frame, sensor_height):
    # Compute the intrinsic camera matrix based on frame dimensions.
    height, width = frame.shape[:2]
    # Example approximation; adjust based on your calibration.
    fx = 600.0
    fy = 600.0
    cx = width / 2.0
    cy = height / 2.0
    camera_matrix = np.array([
        [float(fx), 0.0,       float(cx)],
        [0.0,       float(fy), float(cy)],
        [0.0,       0.0,       1.0]
    ], dtype=np.float32)
    return camera_matrix

def get_fov_y(camera_matrix, frame_height):
    """
    Compute the vertical field of view from focal length for OpenGL rendering
    """
    focal_length_y = camera_matrix[1][1]
    fov_y = np.rad2deg(2 * np.arctan2(frame_height, 2 * focal_length_y))
    return fov_y

def get_matrix44(rvec, tvec):
    """
    Convert the rotation vector and translation vector to a 4x4 matrix
    """
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)
    T = np.eye(4)
    R, jac = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def solvepnp(model_landmarks_list, image_landmarks_list, camera_matrix, frame_width, frame_height): 
    """
    Solve a global rotation and translation to bring the hand model points into the camera space, so that their projected points match the hands. 
    """
    if not model_landmarks_list:
        return []
    
    world_landmarks_list = []
    
    for (model_landmarks, image_landmarks) in zip(model_landmarks_list, image_landmarks_list):
        
        # N x 3 matrix
        model_points = np.float32([[l.x, l.y, l.z] for l in model_landmarks])
        image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in image_landmarks])
        
        if len(model_points) < 4 or len(image_points) < 4:
            continue
        
        world_points = np.copy(model_points)
        
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, None)
        if success:
            world_points = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, None)[0].reshape(-1, 3)
        
        # Store all 3D landmarks
        world_landmarks_list.append(world_points)
    
    return world_landmarks_list

def reproject(world_landmarks_list, image_landmarks_list, 
              camera_matrix, frame_width, frame_height): 
    """
    Perform a perspective projection of 3D points onto the image plane
    and return the projected points.
    """
    reprojection_points_list = []
    reprojection_error = 0.0
    for (world_landmarks, image_landmarks) in zip(world_landmarks_list, image_landmarks_list):
        # Perspective projection by multiplying with the intrinsic matrix
        output = world_landmarks.dot(camera_matrix.T)
        
        # Perspective division
        output[:, 0] /= output[:, 2]
        output[:, 1] /= output[:, 2]
        
        # Store the results into a list for visualization later
        reprojection_points_list.append(output[:, :2])
    
        # Calculate the reprojection error, per point
        image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in image_landmarks])
        
        # Ensure the same number of points by selecting the first n points
        n = min(output.shape[0], image_points.shape[0])
        output_subset = output[:n, :2]
        image_points_subset = image_points[:n]
    
        reprojection_error += np.linalg.norm(output_subset - image_points_subset) / n / len(world_landmarks_list)
    
    return reprojection_error, reprojection_points_list

"""
This is an example main function that displays the video camera and the detection results in 2D landmarks with an OpenCV window.
"""
if __name__ == '__main__':
    # (0) in VideoCapture is used to connect to your computer's default camera
    capture = cv2.VideoCapture(0)
    
    # Initialize previousTime using current time to avoid first FPS error
    previousTime = time.time()
    currentTime = previousTime
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break  # Stop the loop if no frame is retrieved
        
        # resizing the frame for better view
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(720 * aspect_ratio), 720))
        frame = cv2.flip(frame, 1)

        # Converting from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Making predictions
        detection_result = predict(frame)
        
        # Visualize 2D landmarks
        frame = draw_landmarks_on_image(frame, detection_result)
        
        if detection_result and detection_result.hand_world_landmarks and detection_result.hand_landmarks:
            model_landmarks_list = detection_result.hand_world_landmarks[0]
            image_landmarks_list = detection_result.hand_landmarks[0]
            camera_matrix = get_camera_matrix(frame, frame.shape[0])
            world_landmarks_list = solvepnp([model_landmarks_list], [image_landmarks_list], camera_matrix, frame.shape[1], frame.shape[0])
            reprojection_error, reprojection_points_list = reproject(world_landmarks_list, [image_landmarks_list], camera_matrix, frame.shape[1], frame.shape[0])
            for hand_landmarks in reprojection_points_list:
                for l in hand_landmarks:
                    cv2.circle(frame, (int(l[0]), int(l[1])), 3, (0, 0, 255), 2)
        # Optional: else add a status message if no hand is detected
        # else:
        #     cv2.putText(frame, "No hand detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        
        # Displaying FPS on the image
        cv2.putText(frame, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert back to BGR for display and show the image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Detection", frame)
        
        # Break loop on 'Esc' key press
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # When all the process is done
    # Release capture and destroy all windows
    capture.release()
    cv2.destroyAllWindows()

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
import time

app = Flask(__name__)

# Initialize MediaPipe Hand Landmarks and Pose models
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam feed
camera = cv2.VideoCapture(0)

# Initialize frame counter
frame_number = 0


@app.route('/')
def index():
    # Render the homepage
    return render_template('index.html')

# Function to run KMeans
def run_kmeans(X):
    kmeans = KMeans(n_clusters=20, random_state=0).fit(X)

dataForKmeans = []

def generate_frames():
    global frame_number
    while True:
        # Capture frame from the webcam
        success, frame = camera.read()
        if not success:
            break
        else:
            # Increment the frame number
            frame_number += 1

            # Print the image/frame number
            print(f"Processing Frame: {frame_number}")

            # Convert the BGR image to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image for hand landmarks
            hand_results = hands.process(image)

            # Process the image for pose landmarks
            pose_results = pose.process(image)

            # Convert the RGB image back to BGR for OpenCV display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Initialize placeholders for left and right hand landmarks
            pose_landmarks_list = []
            left_hand_landmarks = [(0, 0, 0)] * 21  # 21 landmarks for left hand initialized to (0, 0, 0)
            right_hand_landmarks = [(0, 0, 0)] * 21  # 21 landmarks for right hand initialized to (0, 0, 0)

            # Draw hand landmarks if detected and print 3D coordinates
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                                  hand_results.multi_handedness):
                    
                    hand_label = handedness.classification[0].label[0:]
                    
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Print 3D hand landmark coordinates (x, y, z)
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        hand_landmarks_list = [[np.float16(landmark.x * w), np.float16(landmark.y * h), landmark.z]
                                    for landmark in hand_landmarks.landmark]
                        
                        if hand_label == 'Left':
                            right_hand_landmarks = hand_landmarks_list  # Assign to right hand landmarks
                        elif hand_label == 'Right':
                            left_hand_landmarks = hand_landmarks_list  # Assign to left hand landmarks
                    

            # Print the left hand landmarks (or placeholders if no left hand detected)
            # print("Left hand landmarks:")
            # for idx, (x, y, z) in enumerate(left_hand_landmarks):
            #     print(f"Landmark {idx}: (x: {x}, y: {y}, z: {z:.4f})")

            # Print the right hand landmarks (or placeholders if no right hand detected)
            # print("Right hand landmarks:")
            # for idx, (x, y, z) in enumerate(right_hand_landmarks):
            #     print(f"Landmark {idx}: (x: {x}, y: {y}, z: {z:.4f})")

            # Draw pose landmarks if detected and print 3D coordinates
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))  # Connections

                # Print 3D pose landmark coordinates (x, y, z)
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy, cz = np.float16(landmark.x * w), np.float16(landmark.y * h), landmark.z
                    pose_landmarks_list = [[cx, cy, cz] for landmark in pose_results.pose_landmarks.landmark]
                    # print(f"Pose landmark {idx}: (x: {cx}, y: {cy}, z: {cz:.4f})")
            
            
            hand_pose_landmarks = left_hand_landmarks + right_hand_landmarks + pose_landmarks_list
            hand_pose_landmarks = np.array(hand_pose_landmarks)
            hand_pose_landmarks = hand_pose_landmarks.flatten()


            dataForKmeans.append(hand_pose_landmarks)

            if len(dataForKmeans) == 20:
                start_time = time.time()
                kmeans = KMeans(n_clusters=15, random_state=0).fit(dataForKmeans)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Running time of KMeans: {execution_time:.4f} seconds")
                print('Centers found by scikit-learn:')
                print(kmeans.cluster_centers_)
                for i in range(15):
                    dataForKmeans.pop(0)
                    

            # Encode the frame to be sent to the client
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

@app.route('/video_feed')
def video_feed():
    # Return the response generated by the video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

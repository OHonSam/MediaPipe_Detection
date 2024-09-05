import copy
import itertools
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

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

# Initialize lists to store hand and pose landmarks
right_hand_landmarks_list = []
left_hand_landmarks_list = []
pose_landmarks_list = []
hand_pose_landmarks = []

@app.route('/')
def index():
    # Render the homepage
    return render_template('index.html')

def pre_process_landmark(landmark_list):
    '''
    Hàm preprocess landmark
    '''
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Find the furthest point from the base points
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

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
            # left_hand_landmarks = [(0, 0, 0)] * 21  # 21 landmarks for left hand initialized to (0, 0, 0)
            # right_hand_landmarks = [(0, 0, 0)] * 21  # 21 landmarks for right hand initialized to (0, 0, 0)

            # Draw hand landmarks if detected and print 3D coordinates
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                                  hand_results.multi_handedness):
                    
                    hand_label = handedness.classification[0].label[0:]
                    
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Print 3D hand landmark coordinates (x, y, z)
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        hand_landmarks_list = [[int(landmark.x * w), int(landmark.y * h), landmark.z]
                                    for landmark in hand_landmarks.landmark]
                        '''
                            Tạm thời không quan tâm trái phải
                        '''
                        if hand_label == 'Left':
                            right_hand_landmarks = hand_landmarks_list  # Assign to right hand landmarks
                        elif hand_label == 'Right':
                            left_hand_landmarks = hand_landmarks_list  # Assign to left hand landmarks
                    
            
            # Print the left hand landmarks (or placeholders if no left hand detected)
            # print("Left hand landmarks:")
                pre_process_right_hand_landmark_list = pre_process_landmark(right_hand_landmarks)
                pre_process_left_hand_landmark_list = pre_process_landmark(left_hand_landmarks)

            # for idx, (x, y, z) in enumerate(left_hand_landmarks):
            #     print(f"Landmark {idx}: (x: {x}, y: {y}, z: {z:.4f})")

            #         # Print the right hand landmarks (or placeholders if no right hand detected)
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
                    cx, cy, cz = int(landmark.x * w), int(landmark.y * h), landmark.z
                    pose_landmarks_list = [[cx, cy, cz] for landmark in pose_results.pose_landmarks.landmark]
                    # print(f"Pose landmark {idx}: (x: {cx}, y: {cy}, z: {cz:.4f})")
            
                pre_process_pose_landmark = pre_process_landmark(pose_landmarks_list)

            '''
                Kết hợp right_hand_landmark, left_hand_landmark và pose_landmark nếu như có hand_landmark và pose_landmark được 1 data points
            '''
            if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                hand_pose_landmarks = pre_process_right_hand_landmark_list + pre_process_left_hand_landmark_list + pre_process_pose_landmark
                hand_pose_landmarks = np.array(hand_pose_landmarks)
                print(hand_pose_landmarks.shape)

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

import cv2
import mediapipe as mp
import numpy as np
import os, time

# MediaPipe 모델 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 설정 및 데이터 저장 경로
action = "0"
video_files = [f"LSTM-Practice/video/{action}.mp4"]
# created_time = int(time.time())
seq_length = 5  # 프레임 길이(=윈도우)
save_path = "LSTM-Practice/dataset/"

# 데이터 저장 폴더 생성
os.makedirs(save_path, exist_ok=True)

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    
    # 데이터 시퀀스 초기화
    data_hands_sequence = []
    data_pose_sequence = []
    data_face_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 랜드마크 검출
        results_hands = hands.process(frame)
        results_pose = pose.process(frame)
        results_face_mesh = face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 손 랜드마크 처리
        hands_data = []
        if results_hands.multi_hand_landmarks is not None:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                hands_data.append(joint.flatten())
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if hands_data:
            data_hands_sequence.append(np.mean(hands_data, axis=0))

        # 포즈 랜드마크 처리
        if results_pose.pose_landmarks is not None:
            pose_landmarks = results_pose.pose_landmarks
            pose_joint = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark]).flatten()
            data_pose_sequence.append(pose_joint)
            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 얼굴 랜드마크 처리
        face_data = []
        if results_face_mesh.multi_face_landmarks is not None:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                face_joint = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
                face_data.append(face_joint)
                # 얼굴 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        if face_data:
            data_face_sequence.append(np.mean(face_data, axis=0))

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

            
    # 각 랜드마크 데이터 시퀀스 저장
    def save_data_sequence(data_sequence, label, action):
        full_seq_data = [data_sequence[i:i+seq_length] for i in range(len(data_sequence)-seq_length+1)]
        if full_seq_data:
            np.save(os.path.join(save_path, f'{label}_{action}'), np.array(full_seq_data))

        print(action, np.array(full_seq_data).shape)

    save_data_sequence(data_hands_sequence, "hands", action)
    save_data_sequence(data_pose_sequence, "pose", action)
    save_data_sequence(data_face_sequence, "face", action)

    print(f"Data for action '{action}' saved successfully.")

# 자원 해제
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import os

# 미디어 파이프 모델 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 경로
video_files = ["LSTM-Practice/video/0.mp4"]

action = "0"

# 프레임당 시퀀스 길이
sequence_length = 5

# 데이터 저장 경로
save_path = "LSTM-Practice/dataset/"

for video_file in video_files:
    # 동영상 불러오기
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 미디어 파이프 모델 초기화
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence_count = 0
        frame_sequence = []
        sequence_buffer = np.empty((0, sequence_length, 1668))  # 시퀀스 길이에 맞게 numpy 배열 초기화

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 랜드마크 검출
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # 시퀀스에 추가
            frame_landmarks = np.concatenate([
                np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks]).flatten(),  # 포즈 랜드마크
                np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten(),  # 왼손 랜드마크
                np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten(),  # 오른손 랜드마크
                np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks]).flatten()  # 얼굴 랜드마크
            ])
            frame_sequence.append(frame_landmarks)

            # 시퀀스가 완성되면 저장
            if len(frame_sequence) == sequence_length:
                sequence_count += 1
                sequence_buffer = np.append(sequence_buffer, [frame_sequence], axis=0)
                frame_sequence.clear()

                # 시퀀스 길이만큼 모았으면 numpy 배열로 저장
                if sequence_count == sequence_length:
                    sequence_count = 0
                    np.save(os.path.join(save_path, f'seq_{action}'), sequence_buffer)
                    sequence_buffer = np.empty((0, sequence_length, 1668))  # numpy 배열 다시 초기화
                    print(action, sequence_buffer.shape)

        # 랜드마크 표시
        mp_drawing.draw_landmarks(frame, frame_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, frame_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(frame, frame_landmarks, mp_holistic.POSE_CONNECTIONS)
    
        # 영상을 화면에 표시, 저장
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            # ESC키로 break
            break

        cap.release()

print("Data saved successfully.")
import cv2
import mediapipe as mp
import numpy as np
import os, time

# 미디어 파이프 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 설정
video_files = ["LSTM-Practice/video/2.mp4"]
action = "2"
idx = "3"
created_time = int(time.time())

# 프레임당 시퀀스 길이
sequence_length = 5

# 데이터 저장 경로
save_path = "LSTM-Practice/dataset/"

for video_file in video_files:
    # 동영상 불러오기
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        data_hands = []
        
        # 랜드마크 검출
        results_hands = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results_hands.multi_hand_landmarks is not None:
            for hand_landmarks in results_hands.multi_hand_landmarks:

                # 손의 관절 위치와 가시성 정보 저장할 배열 생성
                joint_hands = np.zeros((21, 4))

                # 모든 관절에 대해 반복
                for j, lm in enumerate(hand_landmarks.landmark):
                    # 관절의 x, y, z 좌표 및 가시성 정보를 배열에 저장
                    joint_hands[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # 관절 간의 각도 계산
                    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint

                    v = v2 - v1 # [20, 3]. 20개 행과 3개 열

                    # 벡터 정규화
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    # radian -> degree
                    angle = np.degrees(angle)

                    # 라벨에 각도 정보 추가
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    # 데이터에 랜드마크와 각도 정보 추가
                    d = np.concatenate([joint_hands.flatten(), angle_label])

                    data_hands.append(d)

                    # 손 랜드마크 그리기
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # 수집한 데이터 저장
    data = np.array(data_hands)
    print(action, data.shape)
    np.save(os.path.join(save_path, f'raw_{action}_{created_time}'), data)

    # 시퀀스 데이터 생성
    seq_length = 5  # 프레임 길이(=윈도우)
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join(save_path, f'seq_{action}_{created_time}'), full_seq_data)

# 사용된 함수, 자원 해제
cap.release()
cv2.destroyAllWindows()

print("Data saved successfully.")
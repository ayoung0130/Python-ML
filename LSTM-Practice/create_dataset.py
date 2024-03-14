import cv2
import mediapipe as mp
import numpy as np
import time, os

# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# MediaPipe pose model 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# # MediaPipe face model 초기화
# mp_face = mp.solutions.face_mesh
# face = mp_face.FaceMesh(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# mp drawing 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 파일 설정
action = "1"
input_video_path = "LSTM-Practice/video/1.mp4"
save_video_path = "LSTM-Practice/dataset/"
idx = 1
created_time = int(time.time())

cap = cv2.VideoCapture(input_video_path)

# 재생할 파일의 넓이와 높이
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# video controller
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(save_video_path, fourcc, 30.0, (int(width), int(height)))

# 동작 수집 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 랜드마크 검출
    results_pose = pose.process(frame)
    results_hands = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    data_hands = []
    data_pose = []

    if results_hands.multi_hand_landmarks:
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

    if results_pose.pose_landmarks:
            
        joint_pose = np.zeros((33, 4))

        for j, lm in enumerate(results_pose.pose_landmarks.landmark):
            joint_pose[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint_pose[[0,0,11,12,13,14,15,16,15,16,15,16,0,0], :3]
        v2 = joint_pose[[11,12,13,14,15,16,17,18,19,20,21,22,23,24], :3]

        v = v2 - v1

        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        angle = np.arccos(np.einsum('nt,nt->n', v[:-1], v[1:])) # [12,]

        angle = np.degrees(angle)

        angle_label = np.array([angle], dtype=np.float32)
        angle_label = np.append(angle_label, idx)

        d = np.concatenate([joint_pose.flatten(), angle_label])

        data_pose.append(d)

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 영상을 화면에 표시, 저장
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        # ESC키로 break
        break

# 수집한 데이터 저장
data = np.array(data_hands)
print(action, data.shape)
np.save(os.path.join(save_video_path, f'raw_{action}_{created_time}'), data)

# 시퀀스 데이터 생성
seq_length = 5
full_seq_data = []
for seq in range(len(data) - seq_length):
    full_seq_data.append(data[seq:seq + seq_length])

full_seq_data = np.array(full_seq_data)
print(action, full_seq_data.shape)
np.save(os.path.join(save_video_path, f'seq_{action}_{created_time}'), full_seq_data)

# 사용된 함수, 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
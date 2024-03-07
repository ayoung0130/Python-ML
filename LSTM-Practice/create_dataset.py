import cv2
import mediapipe as mp
import numpy as np
import time, os

# 수집할 손동작 목록, 시퀀스 길이, 각 동작 할당 시간 설정
actions = ['0', '1', '2']
seq_length = 30
secs_for_action = 4


# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 영상 파일 경로
video_files = ['dataset/0.mp4', 'dataset/1.mp4', 'dataset/2.mp4']

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_files)

# 데이터 저장 디렉토리 생성
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

# 손동작 수집 루프
for video_file in video_files:
    for idx, action in enumerate(actions):
        data = []

        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_file)

        # 시퀀스 수집 시작 시간 기록
        start_time = time.time()

        # 지정된 시간 동안 수집
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            # 종료 조건
            if not ret:
                break

            # 이미지 좌우 반전, 색상 변환
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 손동작 검출
            result = hands.process(img)
            # 색상 변환 원복
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 검출된 동작에 랜드마크 그리기
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:

                    # 손의 관절 위치와 가시성 정보 저장할 배열 생성
                    joint = np.zeros((21, 4))

                    # 모든 관절에 대해 반복
                    for j, lm in enumerate(res.landmark):
                        # 관절의 x, y, z 좌표 및 가시성 정보를 배열에 저장
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # 관절 간의 각도 계산
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
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
                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d)

                    # 이미지에 랜드마크 그리기
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # 화면에 이미지 표시
            cv2.imshow('img', img)

        # 수집한 데이터 저장
        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # 시퀀스 데이터 생성
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
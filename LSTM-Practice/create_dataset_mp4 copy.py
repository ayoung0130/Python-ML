import cv2
import mediapipe as mp
import numpy as np
import os, time

def angle(joint_hands, idx):
    # 관절 간의 각도 계산
    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint  각 관절은 [x, y, z] 좌표로 표현되므로 :3
    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]. 20개 행과 3개 열

    # 벡터 정규화
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
    angle = np.degrees(angle)

    # 라벨에 각도 정보 추가
    angle_label = np.array([angle], dtype=np.float32)
    angle_label = np.append(angle_label, idx)

    return angle_label

# 미디어 파이프 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 설정
action = "0"
idx = 1
video_files = [f"LSTM-Practice/video/{action}.mp4"]
created_time = int(time.time())

# 데이터 저장 경로
save_path = "LSTM-Practice/dataset/"

for video_file in video_files:
    # 동영상 불러오기
    cap = cv2.VideoCapture(video_file)

    hand_data = []
    left_hand_data = []     # 왼손 데이터 배열 생성
    right_hand_data = []    # 오른손 데이터 배열 생성

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame)    # 랜드마크 검출
        results_left_hands = hands.process(frame, handedness='LEFT')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 손 검출시
        if results_hands.multi_hand_landmarks is not None :
            for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness) :

                if len(results_hands.multi_hand_landmarks) == 2:    # 두 손 검출
                    left_joint_hands = np.zeros((21, 4))
                    right_joint_hands = np.zeros((21, 4))

                    for j, lm in enumerate(res.landmark):
                        # 관절의 x, y, z좌표, 가시성 정보를 배열에 저장
                        if handedness.classification[0].label == 'Left':
                            left_joint_hands[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        else:
                            right_joint_hands[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # 데이터에 랜드마크와 각도 정보 추가
                    d_left = np.concatenate([left_joint_hands.flatten(), angle(left_joint_hands, idx)])
                    d_right = np.concatenate([right_joint_hands.flatten(), angle(right_joint_hands, idx)])
                    
                    # 두 d를 합쳐야함(연결x)
                    d_combined = np.array(list(zip(d_left, d_right)))

                    # 데이터 append
                    hand_data.append(d_combined)
                    

                # else :  # 한 손만 검출
                #     if # 왼손 :
                    
                #     else # 오른손:
            
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

        # # 손 검출 x
        # else :
            
        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break
print(d_left)   # 이상함
print(d_right)
# print(hand_data)

# 사용된 함수, 자원 해제
cap.release()
cv2.destroyAllWindows()
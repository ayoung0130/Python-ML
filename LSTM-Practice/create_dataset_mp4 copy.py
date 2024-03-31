import cv2
import mediapipe as mp
import numpy as np
import os, time

def angleHands(joint_hands):
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
    
    return angle_label.flatten()

def anglePose(joint_pose):
    v1 = joint_pose[[0,0,0,0,7,8,9,10,11,12,11,12,11,12,0,0], :3]
    v2 = joint_pose[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
    # 0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    # 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

    v = v2 - v1

    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n', v[:-1], v[1:])) # [15,]

    angle = np.degrees(angle)

    angle_label = np.array([angle], dtype=np.float32)

    return angle_label.flatten()

def setVisibility(x, y, z):
    if x != 0 and y != 0 and z != 0:
        return 1
    elif x == 0 and y == 0 and z == 0:
        return 0
    else:
        return 0.5

# 미디어 파이프 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 설정
action = "기절"
idx = 1
video_files = ["LSTM-Practice/video/155_기절.mp4", "LSTM-Practice/video/988_기절.mp4", "LSTM-Practice/video/5181_기절.mp4"]
seq_length = 30  # 프레임 길이(=윈도우)

# 데이터 저장 경로
save_path = "LSTM-Practice/dataset/"
data = []   # 전체 데이터 저장할 배열 초기화

for video_file in video_files:
    # 동영상 불러오기
    cap = cv2.VideoCapture(video_file)
    created_time = int(time.time())


    # 관절 정보 저장할 넘파이 배열 초기화
    joint_left_hands = np.zeros((21, 4))
    joint_right_hands = np.zeros((21, 4))        
    joint_pose = np.zeros((21, 4))
    joint = np.zeros((21, 12))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame)    # 손 랜드마크 검출
        results_pose = pose.process(frame)      # 포즈 랜드마크 검출
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 손 검출시
        if results_hands.multi_hand_landmarks is not None:
            for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                
                # 손 -> 모든 관절에 대해 반복
                for j, lm in enumerate(res.landmark):
                    if handedness.classification[0].label == 'Left':
                        joint_left_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y, lm.z)]
                    else:
                        joint_right_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y, lm.z)]
                
                # 손 랜드마크 그리기
                if handedness.classification[0].label == 'Left':    # green
                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))
                else: # blue
                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0)))

        # 전체 데이터(joint) 생성
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])
        
        # 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            lm = results_pose.pose_landmarks.landmark[i]
            joint_pose[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # 데이터에 전체 랜드마크,각도값,인덱스 추가 (총 데이터 12*21+15*3+1 = 298개)
        d = np.concatenate([joint.flatten(), angleHands(joint_left_hands), angleHands(joint_right_hands), anglePose(joint_pose)])
        d = np.append(d, idx)
        
        # 전체 데이터를 배열에 추가
        data.append(d)

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
data = np.array(data)
print("data shape:", action, data.shape)

# 시퀀스 데이터 저장
full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
full_seq_data = np.array(full_seq_data)
np.save(os.path.join(save_path, f'seq_{action}_{created_time}'), full_seq_data)
print("seq data shape:", action, full_seq_data.shape)

# 사용된 함수, 자원 해제
cap.release()
cv2.destroyAllWindows()
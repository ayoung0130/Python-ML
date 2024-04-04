import cv2
import mediapipe as mp
import numpy as np
import os, time
from setting import angleHands, anglePose, setVisibility

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
video_files = ["C:/Users/mshof/Desktop/video/155_기절.mp4"]
seq_length = 10  # 프레임 길이(=윈도우)

# 데이터 저장 경로
save_path = "LSTM-Practice/dataset/"

# 전체 데이터 저장할 배열 초기화
data = []

##
# 확인용
left_hand_data = []
right_hand_data = []
pose_data = []

for video_file in video_files:
    # 동영상 불러오기
    cap = cv2.VideoCapture(video_file)
    created_time = int(time.time())
    f = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame)    # 손 랜드마크 검출
        results_pose = pose.process(frame)      # 포즈 랜드마크 검출
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 관절 정보 저장할 넘파이 배열 초기화
        joint_left_hands = np.zeros((21, 4))
        joint_right_hands = np.zeros((21, 4))
        joint_pose = np.zeros((21, 4))
        joint = np.zeros((21, 12))

        # 손 검출시
        if results_hands.multi_hand_landmarks is not None:
            for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                
                # 손 -> 모든 관절에 대해 반복. 한 프레임에 왼손, 오른손 데이터가 0번부터 20번까지 들어감
                for j, lm in enumerate(res.landmark):
                    if handedness.classification[0].label == 'Left':
                        joint_left_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y, lm.z)]
                    else:
                        joint_right_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y, lm.z)]
                
                # 손 랜드마크 그리기
                if handedness.classification[0].label == 'Left':
                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))   # green
                else:
                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0)))   # blue

        # 전체 데이터(joint) 생성, 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])
            joint_pose[j] = [plm.x, plm.y, plm.z, plm.visibility]

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

        ##
        # 데이터 확인용
        if f==20:
            print(f, "번째 프레임:")   
            print("left\n", joint_left_hands)
            print("right\n", joint_right_hands)
            print("pose\n", joint_pose)
            print("left angle\n", angleHands(joint_left_hands))
            print("right angle\n", angleHands(joint_right_hands))
            print("pose angle\n", angleHands(joint_pose), "\n\n")
        f += 1

# 넘파이 배열로 생성
data = np.array(data)
print("data shape: ", action, data.shape)
print("data[20]\n", data[20])

# 시퀀스 데이터 저장
full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
full_seq_data = np.array(full_seq_data)
# np.save(os.path.join(save_path, f'seq_{action}_{created_time}'), full_seq_data)
print("seq data shape:", action, full_seq_data.shape)

# 사용된 함수, 자원 해제
cv2.destroyAllWindows()
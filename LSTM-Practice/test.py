import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
from setting import angleHands, anglePose, setVisibility


actions = ['', '']
seq_length = 10

model = load_model('LSTM-Practice/models/model.h5')

# 미디어 파이프 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame)    # 손 랜드마크 검출
    results_pose = pose.process(frame)      # 포즈 랜드마크 검출
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 관절 정보 저장할 넘파이 배열 초기화
    joint_left_hands = np.zeros((21, 4))
    joint_right_hands = np.zeros((21, 4))        
    joint_pose = np.zeros((21, 4))
    joint = np.zeros((21, 12))

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

        # 전체 데이터(joint) 생성 / 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])
            joint_pose[j] = [plm.x, plm.y, plm.z, plm.visibility]

        # 데이터에 전체 랜드마크,각도값,인덱스 추가 (총 데이터 12*21+15*3+1 = 298개)
        d = np.concatenate([joint.flatten(), angleHands(joint_left_hands), angleHands(joint_right_hands), anglePose(joint_pose)])
                
        seq.append(d)

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        # 모델로 예측 수행
        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        # conf 90% 미만일시 제스처 취하지 않은 것으로 판단
        if conf < 0.9:
            continue

        # 예측된 action label 얻기
        action = actions[i_pred]
        action_seq.append(action)

        # action seq 완료되지 않았다면 계속 진행
        if len(action_seq) < 3:
            continue

        this_action = '?'

        # 마지막 세개의 액션이 모두 같을 때(반복될 때) 유효하다고 판단
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

        for res in results_hands.multi_hand_landmarks:
            cv2.putText(frame, f'{this_action.upper()}', org=(int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('sign language translation', frame)
    if cv2.waitKey(1) == ord('q'):
        break
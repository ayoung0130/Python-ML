import os
import cv2
import numpy as np

def mp4_to_numpy(mp4_file_path, save_path):
    cap = cv2.VideoCapture(mp4_file_path)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame을 numpy 배열로 변환
        frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        frames.append(frame_np)

    cap.release()

    # NumPy 배열로 변환
    video_frames_np = np.array(frames)

    # 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)

    # 파일로 저장
    np.save(os.path.join(save_path, '2.np'), video_frames_np)

# 사용
mp4_file_path = 'KakaoTalk_20240301_145519589.mp4'
save_path = 'LSTM-Practice/dataset'
video_frames = mp4_to_numpy(mp4_file_path, save_path)
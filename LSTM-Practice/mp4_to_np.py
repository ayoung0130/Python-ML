import cv2
import numpy as np

# 동영상 파일 열기
cap = cv2.VideoCapture('0.mp4')

# 동영상의 프레임 수, 프레임의 너비 및 높이 가져오기
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 동영상 프레임을 저장할 빈 numpy 배열을 생성
buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

# 반복문을 사용하여 각 프레임을 읽고 numpy 배열에 저장
fc = 0
ret = True

while (fc < frameCount and ret):
    ret, buf[fc] = cap.read()
    fc += 1

# 동영상 파일 닫기
cap.release()


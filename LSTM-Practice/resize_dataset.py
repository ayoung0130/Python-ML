import cv2, os

video_files = ["C:/Users/mshof/Desktop/video/ID_1/부러지다(정).avi", "C:/Users/mshof/Desktop/video/ID_2/부러지다(정).avi",
               "C:/Users/mshof/Desktop/video/ID_3/부러지다(정).avi", "C:/Users/mshof/Desktop/video/ID_4/부러지다(정).avi",
               "C:/Users/mshof/Desktop/video/ID_5/부러지다(정).avi", "C:/Users/mshof/Desktop/video/ID_6/부러지다(정).avi",
               "C:/Users/mshof/Desktop/video/ID_7/부러지다(정).avi", "C:/Users/mshof/Desktop/video/ID_8/부러지다(정).avi",
               "C:/Users/mshof/Desktop/video/ID_9/부러지다(정).avi", "C:/Users/mshof/Desktop/video/ID_10/부러지다(정).avi",]

# 인덱스 0(가렵다), 1(기절), 2(부러지다)
output_folder = "resized_videos_2"
os.makedirs(output_folder, exist_ok=True)

# 너비, 높이 설정
new_width = 720
new_height = 720

idx = 1

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    output_path = os.path.join(output_folder, os.path.basename(f"{idx}_{os.path.basename(video_file)}"))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # .avi 확장자로 설정
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))   # 30.0은 프레임

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임 크기 조절
        resized_frame = cv2.resize(frame, (new_width, new_height))
            
        # 조절된 프레임을 새로운 동영상에 추가
        out.write(resized_frame)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', resized_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    idx += 1
    
    # 작업 완료 후 해제
    cap.release()
    out.release()

print("영상 크기 조절 및 저장 완료")
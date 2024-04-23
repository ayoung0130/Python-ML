import cv2, os

video_files = ["C:/Users/mshof/Desktop/video/ID_1/가렵다(정).avi"]

output_folder = "resized_videos"
os.makedirs(output_folder, exist_ok=True)

new_width = 640
new_height = 720

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    output_path = os.path.join(output_folder, os.path.basename(video_file))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # .avi 확장자로 설정
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

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
    
    # 작업 완료 후 해제
    cap.release()
    out.release()

print("영상 크기 조절 및 저장 완료")
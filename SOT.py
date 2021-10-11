import cv2 #OpenCV

# 트랙커 객체 생성자 함수 리스트
trackers = [cv2.legacy.TrackerBoosting_create,
            cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.legacy.TrackerTLD_create,
            cv2.legacy.TrackerMedianFlow_create,
            cv2.legacy.TrackerCSRT_create,
            cv2.legacy.TrackerMOSSE_create]

tracker = trackers[5] # 설정된 CSRT tracker
isFirst = True # 처음 실핼할 때를 구분하기 위한 변수

video_src = 0 # 비디오 파일
video_src = "./highway.mp4" # 고속도로에서의 차량 영상
cap = cv2.VideoCapture(video_src) # 비디오 파일 가져오기
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
print('영상의 FPS:', fps)

delay = int(1000/fps) # 프레임에 맞게 영상에 delay 주기
win_name = 'Tracking APIs'

while cap.isOpened(): #객체가 지정한 파일로 정상적으로 초기화되어 있는 동안 실행
    ret, frame = cap.read() # 비디오의 한 프레임씩 읽어 들임
		# 제대로 프레임을 읽으면 ret값이 True, 실패하면 False. fram에 읽은 프레임 반환

    if not ret:
        print('Cannot read video file')
        break
		
    img_draw = frame.copy() #결과 프레임을 copy하여 사용 (원본 frame에 영향X)
		# 새로운 프레임에서 추적 위치 찾기
    ok, bbox = tracker.update(frame) # 성공여부-> ok, 객체의 bbox의 좌표 -> bbox
    (x,y,w,h) = bbox
    if ok: # 추적 성공, 복사된 이미지에 bbox 그려주기 -> 영상으로 표현됨
        cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                      (0,255,0), 2, 1) 
    else : # 추적 실패, 실패문구 표시
        cv2.putText(img_draw, "Tracking fail.", (100,80), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    trackerName = tracker.__class__.__name__ 
    cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (100,20), \
                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow(win_name, img_draw) # 화면 표시
    key = cv2.waitKey(delay) & 0xff # delay 제공
    # 스페이스 바 또는 비디오 파일 최초 실행
    if key == ord(' ') or (video_src != 0 and isFirst):
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False)  # 초기 객체 위치 설정        
else:
    print( "Could not open video")
cap.release() # 마지막으로 오픈한 cap 객체를 해제
cv2.destroyAllWindows() # 생성한 모든 윈도우를 제거

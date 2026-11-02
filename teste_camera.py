import cv2

cam = cv2.VideoCapture(0)   # 0 = webcam padrão

while True:
    ok, frame = cam.read()
    if not ok:
        break
    cv2.imshow("Webcam (aperte Q pra sair)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
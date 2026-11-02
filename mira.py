import cv2
import numpy as np

cam = cv2.VideoCapture(0)

# Faixa de cor do alvo (HSV). Este exemplo pega tons de LARANJA.
cor_baixa = np.array([5, 120, 120])
cor_alta  = np.array([20, 255, 255])

while True:
    ok, frame = cam.read()
    if not ok:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, cor_baixa, cor_alta)
    mascara = cv2.erode(mascara, None, iterations=2)
    mascara = cv2.dilate(mascara, None, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) > 300:   # ignora ruído pequeno
            x, y, w, h = cv2.boundingRect(maior)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
            cv2.putText(frame, f"ALVO ({cx},{cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Mira (Q pra sair)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)

# detector de fundo: aprende o que é "parado" e destaca o que se mexe
fundo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

suave_cx, suave_cy = None, None

def desenhar_mira(frame, cx, cy, raio, t):
    cor = (0, 0, 255)
    for i in range(12):
        a = t * 2.0 + i * (2 * np.pi / 12)
        x1 = int(cx + raio * np.cos(a));        y1 = int(cy + raio * np.sin(a))
        x2 = int(cx + (raio + 8) * np.cos(a));  y2 = int(cy + (raio + 8) * np.sin(a))
        cv2.line(frame, (x1, y1), (x2, y2), cor, 1)
    s, b = raio + 20, max(12, raio // 3)
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        x, y = cx + sx * s, cy + sy * s
        cv2.line(frame, (x, y), (x - sx * b, y), cor, 2)
        cv2.line(frame, (x, y), (x, y - sy * b), cor, 2)
    cv2.circle(frame, (cx, cy), int(raio * 0.7), cor, 2)
    cv2.circle(frame, (cx, cy), int(raio * 0.4), cor, 1)
    L = raio
    cv2.line(frame, (cx - L, cy), (cx - 10, cy), cor, 2)
    cv2.line(frame, (cx + 10, cy), (cx + L, cy), cor, 2)
    cv2.line(frame, (cx, cy - L), (cx, cy - 10), cor, 2)
    cv2.line(frame, (cx, cy + 10), (cx, cy + L), cor, 2)
    cv2.circle(frame, (cx, cy), 3, cor, -1)
    p = int(raio + 18 * abs(np.sin(t * 4)))
    cv2.circle(frame, (cx, cy), p, cor, 1)

def barra_status(frame, texto, cor):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, texto, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

while True:
    ok, frame = cam.read()
    if not ok:
        break
    t = time.time()

    mascara = fundo.apply(frame)                 # o que mudou em relação ao fundo
    mascara = cv2.erode(mascara, None, iterations=2)
    mascara = cv2.dilate(mascara, None, iterations=4)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    alvo = None
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) > 800:         # ignora movimentos pequenos
            alvo = cv2.boundingRect(maior)

    if alvo is not None:
        x, y, w, h = alvo
        cx, cy = x + w // 2, y + h // 2
        raio = max(25, int(max(w, h) * 0.6))
        if suave_cx is None:
            suave_cx, suave_cy = cx, cy
        else:
            suave_cx = int(suave_cx * 0.6 + cx * 0.4)
            suave_cy = int(suave_cy * 0.6 + cy * 0.4)
        desenhar_mira(frame, suave_cx, suave_cy, raio, t)
        barra_status(frame, f"ALVO AUTOMATICO  ({cx},{cy})", (0, 0, 255))
    elif suave_cx is not None:                   # alvo parou: mantém a mira na última posição
        desenhar_mira(frame, suave_cx, suave_cy, raio, t)
        barra_status(frame, f"ALVO PARADO  ({suave_cx},{suave_cy})", (0, 0, 255))
    else:
        barra_status(frame, "VARRENDO - nenhum movimento", (0, 210, 255))

    cv2.imshow("Mira automatica", frame)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    if tecla == ord('r'):                        # solta o alvo e volta a varrer
        suave_cx, suave_cy = None, None

cam.release()
cv2.destroyAllWindows()
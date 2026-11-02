import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)

cor_baixa = None
cor_alta = None
hsv_atual = None

lock_frames = 0          # frames seguidos com alvo à vista
LIMIAR_LOCK = 15         # quantos frames pra considerar "travado"

def escolher_cor(event, x, y, flags, param):
    global cor_baixa, cor_alta
    if event == cv2.EVENT_LBUTTONDOWN and hsv_atual is not None:
        h, s, v = hsv_atual[y, x]
        h, s, v = int(h), int(s), int(v)
        cor_baixa = np.array([max(h - 10, 0), max(s - 70, 40), max(v - 70, 40)])
        cor_alta  = np.array([min(h + 10, 179), 255, 255])

cv2.namedWindow("Mira")
cv2.setMouseCallback("Mira", escolher_cor)

def desenhar_mira(frame, cx, cy, progresso, travado, t):
    cor = (0, 0, 255) if travado else (0, 210, 255)   # vermelho / amarelo

    # anel externo girando (dá o "vida")
    raio = 40
    for i in range(12):
        a = t * 2.0 + i * (2 * np.pi / 12)
        x1 = int(cx + raio * np.cos(a));       y1 = int(cy + raio * np.sin(a))
        x2 = int(cx + (raio + 8) * np.cos(a)); y2 = int(cy + (raio + 8) * np.sin(a))
        cv2.line(frame, (x1, y1), (x2, y2), cor, 1)

    # cantos em L que FECHAM conforme trava
    s = int(70 - 15 * progresso)
    b = 18
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        x, y = cx + sx * s, cy + sy * s
        cv2.line(frame, (x, y), (x - sx * b, y), cor, 2)
        cv2.line(frame, (x, y), (x, y - sy * b), cor, 2)

    cv2.circle(frame, (cx, cy), 28, cor, 2)
    cv2.circle(frame, (cx, cy), 16, cor, 1)
    L = 40
    cv2.line(frame, (cx - L, cy), (cx - 10, cy), cor, 2)
    cv2.line(frame, (cx + 10, cy), (cx + L, cy), cor, 2)
    cv2.line(frame, (cx, cy - L), (cx, cy - 10), cor, 2)
    cv2.line(frame, (cx, cy + 10), (cx, cy + L), cor, 2)
    cv2.circle(frame, (cx, cy), 3, cor, -1)

    # anel pulsando quando travado
    if travado:
        p = int(30 + 18 * abs(np.sin(t * 4)))
        cv2.circle(frame, (cx, cy), p, (0, 0, 255), 1)

def barra_status(frame, texto, cor):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, texto, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

while True:
    ok, frame = cam.read()
    if not ok:
        break
    hsv_atual = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    t = time.time()
    achou = False

    if cor_baixa is not None:
        mascara = cv2.inRange(hsv_atual, cor_baixa, cor_alta)
        mascara = cv2.erode(mascara, None, iterations=2)
        mascara = cv2.dilate(mascara, None, iterations=2)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(maior) > 300:
                achou = True
                x, y, w, h = cv2.boundingRect(maior)
                cx, cy = x + w // 2, y + h // 2
                lock_frames = min(lock_frames + 1, LIMIAR_LOCK)
                progresso = lock_frames / LIMIAR_LOCK
                travado = lock_frames >= LIMIAR_LOCK
                desenhar_mira(frame, cx, cy, progresso, travado, t)
                if travado:
                    barra_status(frame, f"ALVO TRAVADO  ({cx},{cy})", (0, 0, 255))
                else:
                    barra_status(frame, "ALVO DETECTADO - travando...", (0, 210, 255))
        if not achou:
            lock_frames = max(lock_frames - 2, 0)
            barra_status(frame, "PROCURANDO...", (0, 210, 255))
    else:
        barra_status(frame, "Clique no objeto para travar a cor", (0, 210, 255))

    cv2.imshow("Mira", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
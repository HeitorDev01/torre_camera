import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cor_baixa = None
cor_alta = None
hsv_atual = None

# ao clicar, pega a cor daquele ponto e trava nela
def escolher_cor(event, x, y, flags, param):
    global cor_baixa, cor_alta
    if event == cv2.EVENT_LBUTTONDOWN and hsv_atual is not None:
        h, s, v = hsv_atual[y, x]
        h, s, v = int(h), int(s), int(v)
        cor_baixa = np.array([max(h - 10, 0),   max(s - 70, 40), max(v - 70, 40)])
        cor_alta  = np.array([min(h + 10, 179), 255, 255])
        print("Cor travada:", cor_baixa, "->", cor_alta)

cv2.namedWindow("Mira")
cv2.setMouseCallback("Mira", escolher_cor)

# desenha a mira estilo alvo
def desenhar_mira(frame, cx, cy, cor=(0, 255, 0)):
    s, b = 60, 18
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        x, y = cx + sx * s, cy + sy * s
        cv2.line(frame, (x, y), (x - sx * b, y), cor, 2)   # cantos em L
        cv2.line(frame, (x, y), (x, y - sy * b), cor, 2)
    cv2.circle(frame, (cx, cy), 34, cor, 2)                 # círculo externo
    cv2.circle(frame, (cx, cy), 20, cor, 2)                 # círculo interno
    L = 46
    cv2.line(frame, (cx - L, cy), (cx - 10, cy), cor, 2)    # cruz com espaço
    cv2.line(frame, (cx + 10, cy), (cx + L, cy), cor, 2)
    cv2.line(frame, (cx, cy - L), (cx, cy - 10), cor, 2)
    cv2.line(frame, (cx, cy + 10), (cx, cy + L), cor, 2)
    cv2.circle(frame, (cx, cy), 4, cor, -1)                 # ponto central

while True:
    ok, frame = cam.read()
    if not ok:
        break
    hsv_atual = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if cor_baixa is not None:
        mascara = cv2.inRange(hsv_atual, cor_baixa, cor_alta)
        mascara = cv2.erode(mascara, None, iterations=2)
        mascara = cv2.dilate(mascara, None, iterations=2)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(maior) > 300:
                x, y, w, h = cv2.boundingRect(maior)
                cx, cy = x + w // 2, y + h // 2
                desenhar_mira(frame, cx, cy)
                cv2.putText(frame, f"ALVO ({cx},{cy})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Clique no objeto para travar a cor", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Mira", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
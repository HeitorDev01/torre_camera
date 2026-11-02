import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)

# escolhe o melhor rastreador disponível
def criar_tracker():
    for nome in ['TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMIL_create']:
        if hasattr(cv2, nome):
            return getattr(cv2, nome)()
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, nome):
            return getattr(cv2.legacy, nome)()
    raise RuntimeError("Nenhum tracker disponivel")

tracker = None
rastreando = False

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

print("Aperte S pra marcar o alvo | Q pra sair")

while True:
    ok, frame = cam.read()
    if not ok:
        break
    t = time.time()

    if rastreando:
        sucesso, caixa = tracker.update(frame)
        if sucesso:
            x, y, w, h = [int(v) for v in caixa]
            cx, cy = x + w // 2, y + h // 2
            raio = max(25, int(max(w, h) * 0.6))
            desenhar_mira(frame, cx, cy, raio, t)
            barra_status(frame, f"ALVO TRAVADO  ({cx},{cy})", (0, 0, 255))
        else:
            barra_status(frame, "ALVO PERDIDO - aperte S pra remarcar", (0, 210, 255))
    else:
        barra_status(frame, "Aperte S pra marcar o alvo", (0, 210, 255))

    cv2.imshow("Mira", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s'):
        caixa = cv2.selectROI("Mira", frame, False)
        if caixa[2] > 0 and caixa[3] > 0:
            tracker = criar_tracker()
            tracker.init(frame, caixa)
            rastreando = True

cam.release()
cv2.destroyAllWindows()
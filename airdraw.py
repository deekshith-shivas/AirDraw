# airdraw_reverse.py
# ONE finger = draw
# TWO fingers = break
# 'c' = clear, 's' = save, 'q' = quit

import cv2, mediapipe as mp, numpy as np, time, math

COLOR = (0, 180, 0)   # pen color (green)
BRUSH = 18            # thickness
MIN_MOVE = 2.0        # ignore tiny micro-shakes

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def lm_to_px(lm, w, h): return (int(lm.x*w), int(lm.y*h))

def fingers_up(px):
    tips=[8,12,16,20]; pips=[6,10,14,18]
    states=[]
    for t,p in zip(tips,pips):
        states.append(px[t][1] < px[p][1])
    return [False] + states  # [thumb,index,middle,ring,pinky]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

canvas = None
prev = None
smooth = None

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame,1)
    h,w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h,w,3),dtype=np.uint8)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        px = [lm_to_px(pt,w,h) for pt in lm.landmark]
        thumb,idx_up,mid_up,_,_ = fingers_up(px)
        idx_tip = px[8]

        # ---- DRAW MODE: Index up + Middle DOWN ----
        if idx_up and not mid_up:
            cur = idx_tip

            if smooth is None:
                smooth = cur
            else:
                smooth = (int(smooth[0]*0.7 + cur[0]*0.3),
                          int(smooth[1]*0.7 + cur[1]*0.3))

            if prev is None:
                prev = smooth

            if dist(prev,smooth) >= MIN_MOVE:
                cv2.line(canvas, prev, smooth, COLOR, BRUSH, lineType=cv2.LINE_AA)
                prev = smooth

        else:
            # ---- BREAK MODE: Index + Middle both up OR no hand ----
            prev = None
            smooth = None

    else:
        prev=None
        smooth=None

    # merge canvas onto frame
    mask = np.any(canvas!=0,axis=2)
    frame[mask] = canvas[mask]

    cv2.putText(frame,"ONE finger = Draw | TWO fingers = Break | c=clear | s=save | q=quit",
                (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)

    cv2.imshow("AirDraw Reverse",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): break
    if key == ord('c'): canvas[:] = 0
    if key == ord('s'):
        name=f"airdraw_{int(time.time())}.png"
        cv2.imwrite(name,frame)
        print("Saved:",name)

cap.release()
cv2.destroyAllWindows()


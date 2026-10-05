import cv2
import mediapipe as mp
import math

# ==============================
# Camera Setup
# ==============================

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 640)

# ==============================
# MediaPipe Setup
# ==============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==============================
# Keyboard Layout
# ==============================

keys = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE","BACK","ENTER","CLR"]
]

key_w = 45
key_h = 45
margin = 8

typed_text = ""

# ==============================
# Smooth Settings
# ==============================

distance_buffer = []
BUFFER_SIZE = 5
PINCH_THRESHOLD = 35

pinch_state = False
prev_pinch_state = False

smooth_x, smooth_y = 0, 0
SMOOTHING = 0.6

# ==============================
# Main Loop
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    index_x, index_y = 0, 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]

            raw_x = int(index.x * w)
            raw_y = int(index.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            # Smooth cursor
            smooth_x = int(smooth_x * SMOOTHING + raw_x * (1 - SMOOTHING))
            smooth_y = int(smooth_y * SMOOTHING + raw_y * (1 - SMOOTHING))

            index_x, index_y = smooth_x, smooth_y

            cv2.circle(frame, (index_x, index_y), 5, (0,255,0), -1)

            # Distance smoothing
            raw_dist = math.hypot(thumb_x - raw_x,
                                  thumb_y - raw_y)

            distance_buffer.append(raw_dist)
            if len(distance_buffer) > BUFFER_SIZE:
                distance_buffer.pop(0)

            smooth_dist = sum(distance_buffer)/len(distance_buffer)

            pinch_state = smooth_dist < PINCH_THRESHOLD

    # Edge click detection
    is_click = False
    if pinch_state and not prev_pinch_state:
        is_click = True

    prev_pinch_state = pinch_state

    # ==============================
    # Text Box
    # ==============================

    cv2.rectangle(frame, (20,20), (w-20,75), (30,30,30), -1)
    cv2.putText(frame,
                typed_text[-45:],
                (30,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,200),
                2)

    # ==============================
    # Draw Keyboard
    # ==============================

    start_y = 100

    for row in keys:

        row_width = len(row)*key_w + (len(row)-1)*margin
        start_x = (w - row_width)//2

        for key in row:

            width = key_w
            if key == "SPACE":
                width = key_w*3

            x1 = start_x
            y1 = start_y
            x2 = x1 + width
            y2 = y1 + key_h

            hover = x1 < index_x < x2 and y1 < index_y < y2

            if hover:
                color = (40,160,240)
                if is_click:
                    if key == "SPACE":
                        typed_text += " "
                    elif key == "BACK":
                        typed_text = typed_text[:-1]
                    elif key == "ENTER":
                        typed_text += "\n"
                    elif key == "CLR":
                        typed_text = ""
                    else:
                        typed_text += key
            else:
                color = (60,120,200)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (20,20,20), 1)

            text_size = cv2.getTextSize(key,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        2)[0]

            cv2.putText(frame, key,
                        (x1+(width-text_size[0])//2,
                         y1+(key_h+text_size[1])//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2)

            start_x += width + margin

        start_y += key_h + margin

    cv2.imshow("Air Keyboard - Clean Modern UI", frame)

    key_press = cv2.waitKey(1) & 0xFF
    if key_press == 27:   # ESC key
        break

cap.release()
cv2.destroyAllWindows()
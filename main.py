import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter
import random
import traceback  # For better error tracking

# ----------------- Load Model -----------------
print("Loading model...")
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_len = model_dict['feature_len']
    supported_classes = model_dict['classes']
except FileNotFoundError:
    print("Error: model.p not found. Run create_model.py first.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ----------------- Labels Dictionary -----------------
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z',
    26: 'Space',
    27: '.',
    28: 'Confirm',  # Must be shown with left hand for submission
    29: 'Backspace'  # For deleting the last character (right hand)
}

# ----------------- Exam Setup -----------------
target_pool = [
    "LION",
    "TIGER",
    "ELEPHANT",
    "GIRAFFE",
    "ZEBRA",
    "KANGAROO",
    "PENGUIN",
    "HIPPOPOTAMUS",
    "RHINOCEROS",
    "CHEETAH"
]
target_words = random.sample(target_pool, 10)

stage_counter = 0
score = 0
current_target = target_words[stage_counter]
current_input = ""
feedback = ""
submission_mode = False  # True when submission is triggered
submission_triggered_time = 0

# ----------------- Camera & Mediapipe Setup -----------------
print("Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For prediction stability
right_history = []
history_length = 5
min_confidence = 0.3
right_confidence_buffer = []

# For confirmation timing
confirm_start = None
prev_confirm_detected = False

# FPS and crash prevention
last_time = time.time()
frame_count = 0
last_frame_time = time.time()
frame_timeout = 5.0
consecutive_errors = 0
max_consecutive_errors = 10

print("Starting exam. Press 'q' to quit.")
print("Instructions:")
print(" - Use RIGHT hand to form letters; each quick left-hand 'Confirm' gesture adds a letter.")
print(" - To submit your word, show only your left hand with the 'Confirm' sign and hold it for ~2 seconds.")
print(" - If both hands are present in the frame, submission is not triggered.")
print(" - You can submit at any time, even if your input doesn't match the target length.")
print(" - The exam interface always shows 'Show Confirm sign to submit'.")

try:
    while True:
        # Frame timeout handling
        current_time = time.time()
        if current_time - last_frame_time > frame_timeout:
            print("Warning: No frames received for", frame_timeout, "seconds. Resetting camera...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not reopen camera. Exiting.")
                break
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            last_frame_time = time.time()
            continue

        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame. Retrying...")
            consecutive_errors += 1
            if consecutive_errors > max_consecutive_errors:
                print(f"Error: Failed to read frame {max_consecutive_errors} times. Exiting.")
                break
            time.sleep(0.1)
            continue
        consecutive_errors = 0
        last_frame_time = time.time()

        if frame is None or frame.size == 0:
            print("Warning: Invalid frame received. Skipping...")
            continue

        try:
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            display_frame = frame.copy()
        except Exception as e:
            print(f"Error in frame processing: {e}")
            continue

        # Draw instructions
        cv2.putText(display_frame, f"Stage {stage_counter+1}/10", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(display_frame, "Right: sign | Left: confirm", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        right_candidate = None
        right_confidence = 0
        left_confirm_count = 0
        right_confirm_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                try:
                    if not hand_landmarks or not handedness.classification:
                        continue
                    hand_label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Build feature vector
                    data_aux = []
                    x_coords, y_coords, z_coords = [], [], []
                    for lm in hand_landmarks.landmark:
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)
                        z_coords.append(lm.z)
                    if not x_coords:
                        continue
                    min_x, min_y, min_z = min(x_coords), min(y_coords), min(z_coords)
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y

                    for lm in hand_landmarks.landmark:
                        data_aux.extend([
                            lm.x - min_x,
                            lm.y - min_y,
                            lm.z - min_z,
                            lm.x - wrist_x,
                            lm.y - wrist_y
                        ])

                    # Pad or trim to feature_len
                    if len(data_aux) < feature_len:
                        data_aux += [0] * (feature_len - len(data_aux))
                    else:
                        data_aux = data_aux[:feature_len]

                    features = np.array(data_aux).reshape(1, -1)
                    features = scaler.transform(features)
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0]
                    conf = float(prob.max())

                    # Count confirms
                    if int(pred) == 28 and conf >= min_confidence:
                        if hand_label == "Left":
                            left_confirm_count += 1
                        else:
                            right_confirm_count += 1

                    # Right-hand letter candidate
                    if hand_label == "Right" and conf >= min_confidence and int(pred) != 28:
                        right_history.append(pred)
                        right_confidence_buffer.append(conf)
                        if len(right_history) > history_length:
                            right_history.pop(0)
                        if len(right_confidence_buffer) > history_length:
                            right_confidence_buffer.pop(0)
                        right_candidate = Counter(right_history).most_common(1)[0][0]
                        right_confidence = sum(right_confidence_buffer) / len(right_confidence_buffer)

                    # Draw bounding box & label
                    x1 = int(min(x_coords)*W) - 10
                    y1 = int(min(y_coords)*H) - 10
                    x2 = int(max(x_coords)*W) + 10
                    y2 = int(max(y_coords)*H) + 10
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    label_text = labels_dict.get(int(pred), "N/A")
                    if hand_label == "Right":
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{label_text} ({conf:.2f})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    else:
                        if int(pred) == 28:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, "Confirm", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                except Exception as e:
                    print(f"Error processing hand {i}: {e}")
                    traceback.print_exc()
                    continue

        # Determine confirm & hand count
        total_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        confirm_detected = (left_confirm_count == 1 and right_confirm_count == 0)

        # ---------------- Confirm Gesture Handling ----------------
        if confirm_detected and not prev_confirm_detected:
            confirm_start = current_time
            if not submission_mode:
                if right_candidate is not None:
                    try:
                        r = int(right_candidate)
                        if 0 <= r < 26:
                            current_input += labels_dict[r]
                        elif r == 26:
                            current_input += " "
                        elif r == 27:
                            current_input += "."
                        elif r == 29 and current_input:
                            current_input = current_input[:-1]
                    except Exception as e:
                        print(f"Error processing right candidate: {e}")
                right_history.clear()
                right_confidence_buffer.clear()
        elif confirm_detected and prev_confirm_detected:
            if confirm_start is not None:
                # Require 2s hold AND exactly one hand in frame
                if (current_time - confirm_start >= 2.0) and not submission_mode and total_hands == 1:
                    submission_mode = True
                    submission_triggered_time = current_time
                    if current_input.upper() == current_target.upper():
                        feedback = "Correct!"
                        score += 1
                    else:
                        feedback = "Incorrect!"
        else:
            confirm_start = None

        prev_confirm_detected = confirm_detected

        # ---------------- Submission Handling ----------------
        try:
            if submission_mode and (current_time - submission_triggered_time >= 2.0):
                stage_counter += 1
                submission_mode = False
                confirm_start = None
                current_input = ""
                feedback = ""
                if stage_counter < len(target_words):
                    current_target = target_words[stage_counter]
                else:
                    print(f"Exam finished. Score: {score}/{len(target_words)}")
                    print("Results saved to exam_results.txt")
                    break
        except Exception as e:
            print(f"Error in submission handling: {e}")
            submission_mode = False
            submission_triggered_time = None

        # ---------------- Display Exam Interface ----------------
        try:
            exam_window = np.full((250, 800, 3), (230, 240, 255), dtype=np.uint8)
            cv2.rectangle(exam_window, (10, 10), (790, 240), (200, 200, 255), 4)

            cv2.putText(exam_window, f"Stage {stage_counter+1}/10", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (50, 50, 200), 2)
            cv2.putText(exam_window, f"Target: {current_target}", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 100), 2)
            cv2.putText(exam_window, f"Input: {current_input}", (20, 130),
                        cv2.FONT_HERSHEY_COMPLEX, 1.4, (0, 0, 0), 2)
            cv2.putText(exam_window, "Show Confirm sign to submit", (20, 180),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.4, (0, 0, 200), 3)
            if feedback:
                color = (0, 150, 0) if feedback == "Correct!" else (0, 0, 200)
                cv2.putText(exam_window, feedback, (20, 220),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.6, color, 3)

            cv2.imshow('Exam Interface', exam_window)
            cv2.imshow('Sign Language Recognition', display_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("User pressed 'q' to quit.")
                break
        except Exception as e:
            print(f"Error displaying windows: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"Fatal error: {e}")
    traceback.print_exc()
finally:
    print("Cleaning up resources...")
    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass

    try:
        with open("exam_results.txt", "w") as f:
            f.write(f"Score: {score}/{len(target_words)}\n")
            f.write("Target Words: " + " ".join(target_words))
        print(f"Exam finished. Score: {score}/{len(target_words)}")
        print("Results saved to exam_results.txt")
    except Exception as e:
        print(f"Error writing results file: {e}")

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Increase detection confidence for better quality
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=1)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory {DATA_DIR} does not exist")
    exit()

data = []
labels = []
processed_count = 0
skipped_count = 0

# Check for empty dataset
dirs = os.listdir(DATA_DIR)
if not dirs:
    print("Error: No data found. Run collect_imgs.py first.")
    exit()

print(f"Processing images from {len(dirs)} classes...")

for dir_ in sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.listdir(class_path):
        print(f"Warning: No images found in class {dir_}")
        continue

    print(f"Processing class {dir_}...")
    class_processed = 0
    class_skipped = 0

    for img_path in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            skipped_count += 1
            class_skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Process only the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]

            data_aux = []
            x_ = []
            y_ = []
            z_ = []  # Also capture depth for better recognition

            # First collect all coordinates
            for landmark in hand_landmarks.landmark:
                x_ += [landmark.x]
                y_ += [landmark.y]
                z_ += [landmark.z]

            # Normalize coordinates relative to hand bounding box and add relative wrist info
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
                data_aux.append(landmark.z - min(z_))
                data_aux.append(landmark.x - wrist_x)
                data_aux.append(landmark.y - wrist_y)

            # Append to dataset
            data.append(data_aux)
            labels.append(dir_)
            processed_count += 1
            class_processed += 1
        else:
            skipped_count += 1
            class_skipped += 1

    print(f"Class {dir_}: processed {class_processed}, skipped {class_skipped}")

print(f"Total: processed {processed_count} images, skipped {skipped_count} images")

if not data:
    print("Error: No hand landmarks could be detected in any of the images.")
    exit()

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created with {len(data)} samples.")
print("Now run create_model.py to train the model.")
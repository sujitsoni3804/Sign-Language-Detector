import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Updated: Use 30 classes:
# 0-25 for alphabets (A-Z), 26 for Space, 27 for full-stop, 28 for Confirm, 29 for Backspace.
number_of_classes = 30
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z',
    26: 'Space',
    27: '.',
    28: 'Confirm',
    29: 'Backspace'
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    # Create directory if it does not exist
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Skip collection if folder already has images
    if os.listdir(class_dir):
        print(f"Data for class {j}: {labels_dict[j]} already exists, skipping collection for this class.")
        continue

    print('Collecting data for class {}: {}'.format(j, labels_dict[j]))
    # Show special instructions based on the gesture sign.
    if j == 28:
        print("NOTE: For the Confirm sign (class 28), please use your LEFT hand (thumbs-up gesture).")
    elif j == 29:
        print("NOTE: For the Backspace sign (class 29), please use your RIGHT hand.")
    else:
        print("Use your RIGHT hand for alphabet, space, or full-stop signs.")

    # Wait for user to press 'r' to start recording for this class.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror image for intuitive feedback
        cv2.putText(frame, f'Show {labels_dict[j]} sign...! Press "R" to start',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('r'):  # Start recording when 'r' is pressed
            break
        elif key == ord('q'):  # Allow quitting the program
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Countdown before starting to collect images
    for countdown in range(3, 0, -1):
        for _ in range(20):  # Approx. 1 second per countdown number
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Starting in {countdown}...',
                        (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(50)

    # Now collect the dataset for this class
    dataset_size = 100  # Adjust sample count per class if needed
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        # Display progress on the frame
        cv2.putText(frame, f'Collecting {labels_dict[j]}: {counter + 1}/{dataset_size}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the image file
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

    print(f"Finished collecting data for {labels_dict[j]}")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")
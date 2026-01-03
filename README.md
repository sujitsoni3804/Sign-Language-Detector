# Sign Language Detector Project

A Python-based sign language detector that uses computer vision and machine learning techniques to recognize American Sign Language (ASL) gestures. This project collects images for each sign, processes them using MediaPipe to extract hand landmarks, trains a RandomForest classifier on the extracted features, and uses a real-time webcam interface for detection and evaluation.

## Overview

This project is designed to create an end-to-end sign language recognition system. It:
- **Collects images**: For 30 distinct classes (26 alphabets, Space, Full-stop, Confirm, and Backspace), a dataset of 100 images per class is captured using a webcam.
- **Preprocesses data**: Extracts hand landmarks with MediaPipe from the raw images and saves the processed dataset using pickle.
- **Trains a model**: Utilizes a RandomForest classifier with hyperparameter tuning (using GridSearchCV) to classify the sign language gestures.
- **Real-time detection**: Implements an exam interface using OpenCV that processes live webcam feeds to predict and confirm gestures for a user test.

## File Structure

```
Sign-Language-Detector/
├── data/                   # Folder storing collected images (each subfolder for a specific class)
├── outputs/                # Folder for saving model evaluation results (e.g., confusion matrix)
├── Alphabets_signs.jpg     # Sample image of all A to Z alphabets sign used in this project.
├── requirements.txt        # List of Python dependencies
├── data.pickle             # Pickle file containing preprocessed landmark data and labels
├── model.p                 # Trained model saved as a pickle file
├── collect_imgs.py         # Script to capture and save images for each gesture
├── create_dataset.py       # Script to process collected images and extract hand landmarks
├── create_model.py         # Script to train, evaluate, and save the classifier model
└── main.py                 # Script for real-time sign language recognition and exam interface
```

## Project Workflow

### 1. Data Collection  
**Script:** `collect_imgs.py`  

**Functionality:**  
- Opens the webcam and collects 100 images per class.  
- Supports 30 classes (A-Z, Space, '.', Confirm, Backspace).  
- Provides on-screen instructions and countdowns.  
- Saves images in subdirectories under the `data/` folder.

### 2. Dataset Creation & Preprocessing  
**Script:** `create_dataset.py`  

**Functionality:**  
- Reads images from the `data/` folder.  
- Uses MediaPipe to detect hand landmarks.  
- Normalizes and pads/truncates the feature vectors.  
- Aggregates data and labels, and saves them as `data.pickle`.

### 3. Model Training  
**Script:** `create_model.py`  

**Functionality:**  
- Loads processed data from `data.pickle`.  
- Splits the data into training and test sets with stratification.  
- Scales features using `StandardScaler`.  
- Trains a RandomForest classifier with grid search hyperparameter tuning.  
- Evaluates the model's performance (accuracy, classification report, confusion matrix).  
- Visualizes the confusion matrix and saves it under `outputs/`.  
- Saves the best model (and scaler) as `model.p`.

### 4. Real-Time Detection and Exam Interface  
**Script:** `main.py`  

**Functionality:**  
- Loads the trained model and scaler from `model.p`.  
- Opens the webcam and applies real-time predictions using MediaPipe.  
- Uses confidence buffers and history to stabilize predictions.  
- Implements an exam interface that displays target words, current user inputs, and feedback.  
- Provides instructions to the user for using both right hand (letter candidates) and left hand (confirmation gesture).  
- Saves final exam results to a text file.

## Setup and Installation

1. **Set up a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

## How to Run

### Collect Images:

Run the image collection script to gather the dataset (if not already collected):

```bash
python collect_imgs.py
```

### Create Dataset:

Process the collected images to extract hand landmarks:

```bash
python create_dataset.py
```

### Train the Model:

Train the classification model using the processed data:

```bash
python create_model.py
```

### Real-Time Recognition:

Run the main script to launch the sign language recognition system and exam interface:

```bash
python main.py
```

## Dependencies

The project requires the following Python packages:

- OpenCV (cv2)
- MediaPipe
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- pickle (standard library)
- random, time, os, and other standard modules

For a full list, see the `requirements.txt` file.

## Project Details

### Data Collection Script: `collect_imgs.py`

**Purpose:**
Captures images via webcam for each of the 30 sign classes.

**Key Features:**
- Automatic directory creation per class.
- User instructions and countdown timer.
- Hands orientation instructions (left hand for Confirm, right hand for others).

### Dataset Preparation Script: `create_dataset.py`

**Purpose:**
Processes images stored in the `data/` folder, detects hand landmarks using MediaPipe, and prepares a normalized dataset.

**Key Features:**
- Iterates through each image, converting BGR to RGB.
- Handles missing or unreadable images.
- Normalizes landmark coordinates and adds wrist-relative features.
- Saves the final dataset as a pickle file (`data.pickle`).

### Model Training Script: `create_model.py`

**Purpose:**
Trains a RandomForest classifier on the preprocessed dataset.

**Key Features:**
- Splits data into training and testing sets.
- Applies feature scaling using StandardScaler.
- Uses GridSearchCV for hyperparameter tuning.
- Evaluates the model with accuracy score, classification report, and confusion matrix.
- Saves evaluation visualizations and the trained model.
- Saves the best model (and scaler) as `model.p`.

### Real-Time Recognition Script: `main.py`

**Purpose:**
Runs the sign language detector in real-time through the webcam and facilitates an exam interface to test recognition accuracy.

**Key Features:**
- Loads the trained model and scaler.
- Uses MediaPipe to detect hands and extract features from live video frames.
- Implements gesture confirmation using different hand roles (right-hand for letters, left-hand for confirmation).
- Displays a dynamic exam interface with target words, user input, and scoring feedback.
- Saves exam results to a text file.

## Additional Information

### Configuration:
Adjust parameters in the scripts (e.g., number of images per class, confidence thresholds) as needed for your environment.

### Notes:
- Ensure that your webcam is functional before running the scripts.
- Follow the on-screen instructions during data collection and exam execution to obtain accurate results.
- The project is designed for educational purposes and can be extended with additional gestures or improved model architectures.
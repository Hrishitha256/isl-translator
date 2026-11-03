# isl-translator

# Prajna – Indian Sign Language to Text and Speech

**Neural Storm Hackathon 2026 | Problem Statement FT-04: Sign Language to Text/Speech Translator**

Prajna is a real-time computer vision application that translates Indian Sign Language (ISL) numerals into text and speech using a webcam. It runs entirely on an ordinary laptop without requiring a GPU or cloud processing.

**Demo Video:** [Watch the Demo](https://drive.google.com/file/d/1MFErkUkxI9FVIq441kEvpiOT2OJRVdJf/view)

**Repository:** [isl-translator](https://github.com/Hrishitha256/isl-translator)

**Team Prajna:** Hrishitha, Ramya, Saisree

---

## At a Glance

| Feature | Details |
|---|---|
| Recognition | ISL numerals 1–9 |
| Hand Support | Both left and right hands |
| Accuracy | 94.0% on unseen signers |
| Dataset | 16,214 samples from 6 signers |
| Output | Live text, confidence score and speech |
| Platform | Windows laptops with a webcam |
| Processing | Entirely offline, on-device |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Our Solution](#2-our-solution)
3. [How It Works](#3-how-it-works)
4. [Results](#4-results)
5. [Setup and Run](#5-setup-and-run)
6. [Impact and Usefulness](#6-impact-and-usefulness)
7. [Limitations](#7-limitations)
8. [Future Scope](#8-future-scope)
9. [Team](#9-team)

---

## 1. Problem Statement

Many deaf and hard-of-hearing people in India use Indian Sign Language (ISL), but very few hearing people understand it.

At everyday service points such as clinics, banks, offices and classrooms, this communication gap makes even simple exchanges difficult.

For example:
- Sharing a token number at a hospital.
- Communicating an amount at a bank.
- Providing a room number at an office.
- Sharing phone digits during everyday interactions.

**Goal:** Develop a real-time application that recognizes sign language gestures through a camera and instantly converts them into on-screen text and spoken output.

---

## 2. Our Solution

Prajna is a real-time ISL numeral recognition system that:

- Captures hand gestures through a webcam.
- Detects hand landmarks using MediaPipe.
- Recognizes the corresponding ISL numeral using a machine learning model.
- Displays the recognized sign as text.
- Converts the recognized text into spoken audio.

### Key Features

- Supports ISL numerals from 1 to 9.
- Works with either the left or right hand.
- Displays the predicted sign along with a confidence bar.
- Allows users to add spaces, delete characters and clear the text.
- Supports speaking individual recognized signs or the complete sentence.
- Provides both a browser-based interface (`web_app.py`) and a desktop application (`app.py`).

**Current Scope:** This submission focuses on ISL numerals 1–9. We chose a smaller vocabulary that can be recognized reliably during a live demonstration rather than attempting a larger vocabulary with lower reliability.

Adding alphabets or words requires collecting new recordings and retraining the model, rather than rebuilding the entire application.

### Dataset

Since publicly available ISL datasets are limited, we recorded our own dataset.

- 6 different signers.
- Both left-hand and right-hand recordings.
- Approximately 300 samples per sign per signer.
- 16,214 samples in total.
- Only hand-landmark coordinates are stored, not photos or videos.

---

## 3. How It Works

### System Workflow

```text
Webcam
   ↓
MediaPipe Hand Landmarks
   ↓
Feature Extraction (126 Features)
   ↓
Random Forest Classifier
   ↓
Prediction Smoothing
   ↓
Text Generation and Speech Output
```

### Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| MediaPipe | Hand landmark detection |
| OpenCV | Webcam capture |
| scikit-learn | Machine learning model |
| NumPy | Numerical operations |
| pandas | Dataset processing |
| Flask | Browser-based interface |
| pyttsx3 | Desktop speech synthesis |
| Browser Speech Synthesis | Browser-based speech output |

### 3.1 Webcam and Hand Landmark Detection

OpenCV captures webcam frames, while MediaPipe Hands detects up to two hands.

Each hand is represented by 21 landmarks, with three coordinates (x, y, z) per landmark.

Instead of processing raw images, the system works with these numerical coordinates, making it lightweight enough to run on a laptop CPU.

### 3.2 Feature Extraction (`features.py`)

Each hand's landmarks are measured relative to the wrist and normalized using hand size.

This helps make the recognition less dependent on:

- Hand position in the frame.
- Distance from the camera.
- Differences in hand size.

The left and right hands have separate fixed feature slots.

Therefore, each frame contains:

**2 hands × 21 landmarks × 3 coordinates = 126 features**

If a hand is not visible, its corresponding feature values are filled with zeros.

### 3.3 Classification (`train.py`)

We use a **Random Forest classifier with 100 trees** from scikit-learn.

For every frame, the model predicts the corresponding sign and generates a confidence score.

The confidence score:
- Determines the confidence bar displayed in the interface.
- Helps decide whether a prediction should be accepted.

### 3.4 Prediction Smoothing (`app.py`)

Predictions from individual frames can fluctuate, even when the signer is holding the same gesture.

To avoid incorrect or repeated predictions, we use a smoothing mechanism.

A sign is accepted only when the same prediction remains above the confidence threshold for approximately 12 consecutive frames.

To enter the same sign twice, the signer briefly lowers their hand out of the camera's view between the two signs.

### 3.5 Text Assembly and Speech

Once a sign is accepted, it is added to the text displayed on the screen.

The application supports:
- Adding spaces between entries.
- Deleting the previous character.
- Clearing the complete text.
- Speaking individual recognized signs.
- Speaking the entire sentence.

The browser application uses the browser's built-in speech synthesis, while the desktop application uses `pyttsx3`.

In the desktop application, speech processing runs in the background so that video capture and recognition do not freeze.

**Consistency:** The same feature extraction function (`features.py`) is used during data collection, model training and live prediction. This ensures that all three stages use identical features.

---

## 4. Results

### Model Evaluation

We evaluated our models using **Leave-One-Person-Out Cross-Validation (LOPO)**.

In this approach:
1. The model is trained using recordings from five signers.
2. It is tested on the sixth signer, whom it has never encountered.
3. This process is repeated for all six signers.

This evaluation provides a better estimate of performance on unseen people than a random train-test split, where highly similar consecutive frames from the same recording may appear in both sets.

### Overall Accuracy

| Model | Accuracy on Unseen Signers |
|---|---:|
| Random Forest (100 trees) | **94.0%** |
| SVM (RBF Kernel) | 92.0% |

Random Forest achieved the higher overall accuracy and was selected for the final application.

### Accuracy for Each Held-Out Signer

| Held-Out Signer | Random Forest | SVM |
|---|---:|---:|
| Archana | 96.4% | 97.4% |
| Hrishitha | 99.1% | 94.1% |
| Ramya | 86.7% | 90.2% |
| Saisree | 85.4% | 74.3% |
| Shivani | 97.8% | 97.5% |
| Vikas | 98.7% | 98.6% |
| **Overall** | **94.0%** | **92.0%** |

---

## 5. Setup and Run

### Requirements

- Windows laptop.
- Webcam.
- Python 3.11.
- Internet connection for initial dependency installation.

**Note:** Python 3.11 is recommended because MediaPipe does not support some newer Python versions.

### Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/Hrishitha256/isl-translator.git
```

**Step 2: Navigate to the project directory**

```bash
cd isl-translator
```

**Step 3: Create a virtual environment**

```bash
py -3.11 -m venv venv
```

**Step 4: Activate the virtual environment**

```bash
venv\Scripts\activate
```

**Step 5: Upgrade pip**

```bash
python -m pip install --upgrade pip
```

**Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

### Run the Application

A trained model is already included in the repository at `models/model.pkl`, so there is no need to train the model before running the application.

**Option 1: Browser Application (Recommended)**

```bash
python web_app.py
```

Open the following address in Chrome or Edge:

```text
http://127.0.0.1:5000
```

Click once on the webpage to allow browser audio playback.

**Option 2: Desktop Application**

```bash
python app.py
```

### Collect Your Own Dataset

To record your own sign language samples:

```bash
python collect.py --name yourname
```

### Retrain the Model

After collecting additional data, run:

```bash
python train.py
```

The training script uses all the available files in the `data` folder.

### Tips for Best Results

- Sit approximately an arm's length away from the webcam.
- Use soft and even lighting.
- Prefer a plain background.
- Keep the entire hand visible inside the camera frame.
- Hold each sign steadily for approximately half a second.

---

## 6. Impact and Usefulness

Prajna aims to make everyday communication easier between ISL users and people who do not understand sign language.

Potential application areas include:

- **Hospitals:** Communicating token numbers, room numbers and other numerical information.
- **Banks:** Sharing amounts and numerical details.
- **Offices:** Communicating room numbers and other basic information.
- **Classrooms:** Supporting simple numerical interactions.

### Key Benefits

**1. Low Cost and Easy Setup**

Requires only a laptop and webcam. No GPU, specialized hardware or cloud service is needed.

**2. Privacy-Friendly Processing**

Recognition happens entirely on the laptop. The application does not save camera images or videos, and the recorded training dataset contains only hand-landmark coordinates.

**3. Real-Time Communication**

Recognized signs are displayed as text and can be converted into spoken audio immediately.

**4. Expandable Architecture**

The same recognition pipeline can be extended by recording additional signs and retraining the model.

Prajna is currently a working prototype focused on numerals 1–9. It has been evaluated using six signers but has not yet been tested in real-world environments such as hospitals or banks.

---

## 7. Limitations

The current prototype has the following limitations:

- Recognizes only ISL numerals 1–9.
- Supports static hand poses rather than motion-based signs.
- Has been trained and tested using data from only six signers.
- Performs best under lighting and camera conditions similar to those used during data collection.
- Has not yet been tested on mobile devices.
- Does not currently support complete ISL sentences or conversational translation.

---

## 8. Future Scope

### 8.1 Mobile Application

Our primary next step is to make Prajna accessible through mobile devices.

Currently, Python handles webcam capture and recognition on a laptop. Our proposed approach is to move the complete recognition pipeline into the mobile browser, without requiring a server.

| Component | Current Implementation | Proposed Mobile Implementation |
|---|---|---|
| Camera | OpenCV | Browser Camera API |
| Hand Landmarks | MediaPipe for Python | MediaPipe Hands for Web |
| Classifier | Random Forest | Small Neural Network |
| Model Deployment | Python | TensorFlow.js |
| Smoothing and Text | Python | JavaScript |

**Proposed Approach:**

- Run the entire recognition pipeline inside the mobile browser.
- Reuse the existing dataset and feature definitions.
- Convert the trained classifier into a TensorFlow.js-compatible model.
- Process camera input directly on the device.

This approach avoids the need to record the dataset again and keeps camera processing on the phone.

An alternative approach would be to send camera frames to a hosted Python server. However, this would require hosting, introduce additional latency and transmit camera images outside the device.

*Note: The mobile implementation is a planned extension and has not yet been developed.*

### 8.2 More Signs and Better Recognition

Future improvements include:

- **ISL Alphabet Recognition:** Extend the vocabulary to recognize individual alphabets.
- **Word and Phrase Recognition:** Support commonly used words and phrases.
- **Motion-Based Signs:** Explore sequence models such as LSTM or Transformers to recognize gestures involving movement.
- **Larger Dataset:** Collect recordings from more signers and under different lighting conditions.
- **Improved Features:** Explore additional features such as finger angles to improve recognition.

---

## 9. Team

| Team Member | Contribution |
|---|---|
| Saisree | Data Collection and Model Development |
| Hrishitha | Real-Time Recognition Pipeline |
| Ramya | Speech Integration, User Interface and Documentation |

---

**Prajna – Making Everyday Communication More Accessible Through Technology.**
Saisree	Data and model
Hrishitha	Real-time pipeline
Ramya	Speech, interface and documentation


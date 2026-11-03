# isl-translator

Prajna: Indian Sign Language to Text and Speech
Neural Storm Hackathon 2026 | Problem Statement FT-04: Sign Language to Text/Speech Translator
Prajna is a real-time computer vision application that reads Indian Sign Language (ISL) numerals from a webcam and turns them into on-screen text and spoken audio. It runs on an ordinary laptop with no GPU, and the recognition happens entirely on the laptop.
Demo video: [Watch the demo] https://drive.google.com/file/d/1MFErkUkxI9FVIq441kEvpiOT2OJRVdJf/view
Repository: https://github.com/Hrishitha256/isl-translator
Team Prajna: Hrishitha, Ramya, Saisree

At a glance
Recognizes	ISL numerals 1-9, with either hand
Accuracy	94.0% on signers the model has never seen (leave-one-person-out, 6 signers)
Data	16,214 samples recorded by 6 signers (landmarks only, no photos)
Output	Live text, a confidence display, and spoken audio
Runs on	Windows laptops with a webcam

Screenshots: Refer docs

Contents: 
1. Problem statement 2. Our solution 3. How it works 4. Results 5. Setup and run 6. Impact and usefulness 7. Limitations 8. Future scope 9. Team

1. Problem statement
Many deaf and hard-of-hearing people in India use Indian Sign Language, but very few hearing people know it. At everyday service points such as clinics, banks, offices and classrooms, this gap makes even simple exchanges (a token number, an amount, a room number) difficult.
Goal: a real-time application that reads gestures from a camera and instantly produces on-screen text and spoken output.

2. Our solution
Prajna watches the signer through a webcam, detects the hands, recognizes the sign, builds a sentence, and speaks it aloud. It shows the detected sign with a confidence bar, lets the user add a space, delete or clear the text, and speaks each recognized sign or the whole sentence. It comes as a browser interface (web_app.py) and a desktop window (app.py).
Scope of this submission: the ISL numerals 1 to 9, signed with either the left or the right hand. We chose a small vocabulary that works reliably over a larger one that might fail in a live demo. Adding letters or words needs new recordings and a retrain, not new code (see Future scope).
Dataset: public ISL data is limited, so we recorded our own. Six signers recorded every sign with the right hand and the left hand, about 300 samples per sign per signer, 16,214 samples in total. The files contain only hand-landmark numbers, no photos or video.

3. How it works
Webcam → MediaPipe hand landmarks → Features (126 numbers) → Random Forest classifier → Smoothing → Text and speech
Tech stack: Python 3.11, MediaPipe, OpenCV, scikit-learn, NumPy, pandas, Flask, pyttsx3, browser speech synthesis.
•	Webcam and hand landmarks: OpenCV captures the camera frames and MediaPipe Hands finds up to two hands, giving 21 points (x, y, z) for each hand. The system works on these coordinates and not on raw images, so it is light enough to run on a laptop CPU.
•	Feature extraction (features.py): Each hand’s points are measured from the wrist and scaled by hand size, so a sign looks the same wherever the hand is in the frame and however far it is from the camera. The left and right hands each have a fixed slot, which gives 126 numbers per frame (2 hands × 21 points × 3 values). A hand that is not visible is filled with zeros.
•	Classifier (train.py): A scikit-learn Random Forest with 100 trees predicts the sign and a confidence score for every frame. The confidence drives the confidence bar and decides whether a prediction is accepted.
•	Smoothing (app.py): A prediction from a single frame can flicker, so a sign is accepted only when the same prediction stays above the confidence threshold for about 12 frames in a row. To enter the same sign twice, the signer lowers the hand out of view in between, which prevents accidental repeats.
•	Text assembly and speech: Accepted signs are added to the sentence on screen. The browser app speaks the text with the browser’s built-in voice, and the desktop app uses pyttsx3. In the desktop app, speech runs in the background so the video does not freeze.
The same feature function (features.py) is used for recording data, training and live prediction, so all three use identical features.

4. Results
We evaluate with leave-one-person-out cross-validation: the model is trained on five people and tested on the sixth, who it has never seen, and this is repeated for every person. This measures how the system behaves for a new signer, the realistic situation, and avoids the inflated scores that a random split gives (consecutive frames of one recording are almost identical).
Model	Overall accuracy on unseen signers
Random Forest (100 trees)	94.0%
SVM (RBF kernel)	92.0%
Random Forest scored higher overall, so it was selected.
Accuracy for each held-out signer
Held-out signer	Random Forest	SVM
archana	96.4%	97.4%
hrishitha	99.1%	94.1%
ramya	86.7%	90.2%
saisree	85.4%	74.3%
shivani	97.8%	97.5%
vikas	98.7%	98.6%

5. Setup and run
Requirements: Windows, a webcam, and Python 3.11 (MediaPipe does not support the newest Python versions).
git clone https://github.com/Hrishitha256/isl-translator.git
cd isl-translator
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
A trained model is included (models/model.pkl), so you can run the app straight away.
Run the browser app (recommended): python web_app.py, then type 127.0.0.1:5000 into your browser’s address bar (Chrome or Edge). Click once on the page so the browser allows sound.
Record your own data: python collect.py --name yourname
Retrain the model: python train.py (uses every file in the data folder)
Tips for the best results: sit about an arm’s length from the camera, use soft even lighting and a plain background, keep the whole hand in the frame, and hold each sign steady for about half a second.

6. Impact and usefulness
Prajna targets everyday service points such as a hospital reception, a bank counter, an office desk or a classroom, where a signer needs to share a number (a token number, an amount, a room number or a phone digit) with someone who does not know ISL.
•	Low cost and easy to set up: it needs only a laptop and a webcam. There is no GPU, no special hardware and no cloud service, because recognition runs on the laptop itself.
•	Privacy: the app does not save camera images or video. The data we recorded for training contains only hand-landmark coordinates.
•	Real-time output: the recognized sign appears on screen as text and can be spoken aloud straight away.
•	Easy to extend: the same pipeline can learn new signs from new recordings and a retrain, so Prajna can grow into a fuller ISL translator.
Prajna is a working prototype for the numerals 1 to 9. It has been tested with six signers and has not yet been tried in a real clinic or bank.

7. Limitations
Prajna currently recognizes only the numerals 1 to 9 from single-frame static poses, so signs that need motion are not supported. It was trained and tested on six signers only, works best in good lighting at a camera distance similar to the recordings, and has not yet been tested on phones.

8. Future scope
Mobile version (our main next step). Prajna currently runs on a laptop: Python reads the webcam and runs the recognition, so a phone cannot use its own camera yet. We plan to move the whole pipeline into the phone’s browser. This is a plan and has not been built yet.
Approach: run everything in the phone’s browser, with no server.
Part	Today (laptop)	Mobile plan
Camera	OpenCV	Browser camera API
Hand landmarks	MediaPipe for Python	MediaPipe Hands for the web
Classifier	Random Forest	Small neural network on the same 126 features, exported to TensorFlow.js
Smoothing and text	Python	Same logic in JavaScript
Why this approach: the recorded dataset and the feature definition can be reused, so no recordings need to be redone, and the camera video would stay on the phone.
Alternative: send camera frames to a hosted Python server. This reuses all the current code, but it needs hosting, adds delay, and sends camera images off the device.
More signs and better recognition
•	Alphabet, words and phrases: record new signs with collect.py and retrain.
•	Motion signs: a sequence model (LSTM or Transformer) on short windows of landmarks.
•	More signers and lighting conditions, plus extra features such as finger angles.

9. Team
Name	Role
Saisree	Data and model
Hrishitha	Real-time pipeline
Ramya	Speech, interface and documentation


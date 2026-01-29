import multiprocessing as mp
import numpy as np
import cv2
import mediapipe as mp_pipe
import onnxruntime as ort
from collections import deque

class AIWorker(mp.Process):
    """
    Background process that handles heavy AI inference.
    Decoupled from the main video loop to ensure 60 FPS video.
    """
    def __init__(self, frame_queue, result_queue, model_path):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model_path = model_path
        self.daemon = True 

    def extract_landmark_features(self, hand_landmarks, handedness):
        """Extracts 63 landmark points (x,y,z) and normalizes them"""
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        # Mirror effect for right hand to match training data
        if handedness.classification[0].label == "Right":
            landmarks[:, 0] = 1 - landmarks[:, 0]
        return landmarks.flatten().reshape(1, -1)

    def run(self):
        # --- 1. SETUP (Runs Once) ---
        try:
            # Load the optimized ONNX model (Cpu optimized)
            ort_session = ort.InferenceSession(self.model_path)
            input_name = ort_session.get_inputs()[0].name
            print(f"✅ AI Worker Loaded Model: {self.model_path}")
        except Exception as e:
            print(f"❌ AI Worker CRITICAL FAILURE: {e}")
            return

        # Initialize MediaPipe (Optimized for speed)
        mp_hands = mp_pipe.solutions.hands
        hands = mp_hands.Hands(
            min_detection_confidence=0.5, # Lowered to 0.5 for easier detection
            min_tracking_confidence=0.5, 
            max_num_hands=1
        )

        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        label_mapping = {i: label for i, label in enumerate(class_labels)}

        # Stabilization: Requires 3 matching frames to confirm a letter
        stabilization_window = deque(maxlen=3) 
        predicted_sentence = ""
        last_pred_char = ""
        
        print("✅ AI Worker Ready & Waiting for frames...")

        # --- 2. FAST LOOP ---
        while True:
            try:
                # Non-blocking get (or minimal wait) to keep loop tight
                frame = self.frame_queue.get()
            except:
                continue
            
            # Pre-processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            confidence = 0.0
            two_hands_detected = False

            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) > 1:
                    two_hands_detected = True
                else:
                    # Single Hand Logic
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0]
                    
                    # 1. Feature Extraction
                    feats = self.extract_landmark_features(hand_landmarks, handedness).astype(np.float32)
                    
                    # 2. ONNX Inference (The fast part)
                    outputs = ort_session.run(None, {input_name: feats})
                    prediction = outputs[0][0] # Raw logits
                    
                    # Calculate Confidence
                    exp_preds = np.exp(prediction - np.max(prediction))
                    softmax_preds = exp_preds / exp_preds.sum()
                    idx = np.argmax(softmax_preds)
                    
                    confidence = float(softmax_preds[idx])
                    label = label_mapping.get(idx, "?")

                    # 3. Stabilization Logic
                    stabilization_window.append(label)
                    
                    # Only update text if we have 3 identical consecutive frames
                    if stabilization_window.count(label) == 3:
                        if label != last_pred_char:
                            last_pred_char = label
                            
                            # Valid Character Check
                            if label not in ["nothing", "del", "space", "unknown", "?"]:
                                predicted_sentence += label
                            elif label == "space":
                                predicted_sentence += " "
                            elif label == "del":
                                predicted_sentence = predicted_sentence[:-1]

            # --- 3. SEND RESULT ---
            # Send latest state to Main App. 
            # We use get_nowait() first to ensure queue never fills up/lags.
            if not self.result_queue.empty():
                try: self.result_queue.get_nowait()
                except: pass
            
            self.result_queue.put((predicted_sentence, confidence, two_hands_detected))
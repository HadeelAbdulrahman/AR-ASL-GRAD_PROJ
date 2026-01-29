import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import tf2onnx

# --- CONFIGURATION ---
DATASET_FILE = "asl_mediapipe_keypoints_dataset.csv" # Your existing dataset
OUTPUT_ONNX = "mlp_model.onnx"                       # The file your App uses
# ---------------------

def train_fine_tuned_model():
    print(f"📊 Loading dataset: {DATASET_FILE}...")
    
    if not os.path.exists(DATASET_FILE):
        print("❌ Error: Dataset CSV not found!")
        return

    # 1. Load Data
    df = pd.read_csv(DATASET_FILE)
    
    # Check if data looks correct (Landmarks + Label)
    # Assuming column 0 is 'label' and rest are coordinates
    X = df.iloc[:, 1:].values.astype('float32')
    y = df.iloc[:, 0].values
    
    print(f"   -> Found {len(df)} samples")
    print(f"   -> Found {len(np.unique(y))} unique classes: {np.unique(y)}")

    # 2. Encode Labels (A, B, C -> 0, 1, 2)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    
    # Save the label mapping so you can verify it matches your App's list
    print("   -> Class Mapping:", dict(zip(range(len(encoder.classes_)), encoder.classes_)))

    # 3. Split Data (80% Train, 20% Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 4. Define the MLP Architecture (Lightweight & Fast)
    # Matches the input size of MediaPipe landmarks (21 points * 3 coords = 63 values)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2), # Prevents overfitting
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train
    print("\n🧠 Starting Training...")
    # Early stopping prevents wasting time if model stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,           # Quick training
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Training Complete! Validation Accuracy: {acc*100:.2f}%")

    # 7. Convert to ONNX (The Crucial Step for Speed)
    print(f"\n🔄 Converting to ONNX ({OUTPUT_ONNX})...")
    
    # Convert Keras model to ONNX
    spec = (tf.TensorSpec((None, 63), tf.float32, name="input"),)
    output_path = OUTPUT_ONNX
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    
    print(f"🚀 SUCCESS: '{OUTPUT_ONNX}' saved. Restart your Main App to use it.")

if __name__ == "__main__":
    train_fine_tuned_model()
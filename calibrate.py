import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from fer import FER
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
import threading

print("Initializing FER and camera...")

# Initialize FER
fer = FER()

# --- Webcam detection helper ---
def list_webcams(max_devices=10):
    """Scan for available webcams (up to max_devices)."""
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

print("Scanning for available webcams...")
webcams = list_webcams()

if not webcams:
    print("❌ No webcam detected.")
    exit(1)

if len(webcams) > 1:
    print("\nMultiple webcams detected. Select one:")
    for i, cam_index in enumerate(webcams):
        print(f"{i+1}. Camera {cam_index}")
    try:
        choice = int(input("Enter choice: ").strip()) - 1
        if not 0 <= choice < len(webcams):
            print("Invalid choice. Defaulting to first webcam.")
            choice = 0
    except ValueError:
        print("Invalid input. Defaulting to first webcam.")
        choice = 0
    cam_index = webcams[choice]
else:
    cam_index = webcams[0]

# Initialize camera
cap = cv2.VideoCapture(cam_index)


# --- Per-Person Calibration System ---
class PersonCalibrationManager:
    """
    Manages individual calibration models for each person.
    Each person gets their own trained model.
    """
    def __init__(self):
        self.models = {}  # {person_name: keras.Model}
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.training_data = {}  # {person_name: {'X': [], 'y': []}}
        self.model_dir = "person_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def build_model(self, input_dim=7, num_classes=7):
        """Build a small neural network for emotion calibration."""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def add_training_sample(self, person_name, emotion_vector, true_emotion_idx):
        """Add a training sample for a specific person."""
        if person_name not in self.training_data:
            self.training_data[person_name] = {'X': [], 'y': []}
        
        self.training_data[person_name]['X'].append(emotion_vector)
        self.training_data[person_name]['y'].append(true_emotion_idx)
    
    def get_sample_count(self, person_name):
        """Get the number of samples collected for a person."""
        if person_name not in self.training_data:
            return 0
        return len(self.training_data[person_name]['X'])
    
    def get_emotion_counts(self, person_name):
        """Get count of samples per emotion for a person."""
        if person_name not in self.training_data:
            return {emotion: 0 for emotion in self.emotion_labels}
        
        counts = {emotion: 0 for emotion in self.emotion_labels}
        for idx in self.training_data[person_name]['y']:
            counts[self.emotion_labels[idx]] += 1
        return counts
    
    def train_person_model(self, person_name):
        """Train a model for a specific person."""
        if person_name not in self.training_data:
            print(f"❌ No training data for {person_name}")
            return False, "No training data"
        
        data = self.training_data[person_name]
        if len(data['X']) < 10:
            return False, f"Not enough samples ({len(data['X'])}). Need at least 10."
        
        X_train = np.array(data['X'], dtype=np.float32)
        y_train = np.array(data['y'], dtype=np.int32)
        
        print(f"\n📊 Training model for {person_name} ({len(X_train)} samples)...")
        
        model = self.build_model()
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=4,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
            ]
        )
        
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        
        print(f"   ✅ Training complete! Loss: {final_loss:.4f}, Accuracy: {final_acc:.2%}")
        
        self.models[person_name] = model
        
        # Save model
        safe_name = self._safe_filename(person_name)
        model_path = os.path.join(self.model_dir, f"{safe_name}_model.keras")
        model.save(model_path)
        
        # Save metadata
        metadata = {
            'person_name': person_name,
            'training_samples': len(X_train),
            'final_accuracy': float(final_acc),
            'emotion_labels': self.emotion_labels
        }
        metadata_path = os.path.join(self.model_dir, f"{safe_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, f"Accuracy: {final_acc:.2%}"
    
    def _safe_filename(self, person_name):
        """Convert person name to safe filename."""
        safe = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
        return safe.replace(' ', '_').lower()


# --- GUI Calibration Application ---
class CalibrationGUI:
    def __init__(self, root, cap, fer):
        self.root = root
        self.cap = cap
        self.fer = fer
        self.calibration_manager = PersonCalibrationManager()
        
        self.person_name = None
        self.current_emotion_idx = 0
        self.sample_count = 0
        self.is_capturing = False
        self.latest_frame = None
        self.latest_emotions = None
        
        # Emotion sequence (optimized for good coverage)
        self.emotion_sequence = [
            'neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust',
            'neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust',
            'neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust'
        ] * 8  # 168 samples total (24 per emotion)
        
        self.setup_gui()
        self.start_camera_thread()
        self.update_video_feed()
        
    def setup_gui(self):
        """Setup the GUI interface."""
        self.root.title("Emotion Calibration System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🎭 Emotion Calibration System",
            font=('Arial', 24, 'bold'),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        title_label.pack()
        
        # Person name input (only shown at start)
        self.name_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.name_frame.pack(pady=20)
        
        tk.Label(
            self.name_frame,
            text="Enter your name:",
            font=('Arial', 14),
            bg='#2b2b2b',
            fg='#ffffff'
        ).pack()
        
        self.name_entry = tk.Entry(self.name_frame, font=('Arial', 14), width=20)
        self.name_entry.pack(pady=10)
        
        tk.Button(
            self.name_frame,
            text="Start Calibration",
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            command=self.start_calibration,
            padx=20,
            pady=10
        ).pack()
        
        # Video feed
        self.video_frame = tk.Frame(self.root, bg='#1a1a1a', relief=tk.SUNKEN, bd=2)
        self.video_frame.pack(pady=10)
        
        self.video_label = tk.Label(self.video_frame, bg='#000000')
        self.video_label.pack()
        
        # Main calibration interface (hidden initially)
        self.calib_frame = tk.Frame(self.root, bg='#2b2b2b')
        
        # Current emotion display
        self.emotion_label = tk.Label(
            self.calib_frame,
            text="Emotion: ",
            font=('Arial', 32, 'bold'),
            bg='#2b2b2b',
            fg='#FFD700'
        )
        self.emotion_label.pack(pady=20)
        
        # Progress bar
        progress_frame = tk.Frame(self.calib_frame, bg='#2b2b2b')
        progress_frame.pack(pady=10, fill=tk.X, padx=50)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=600,
            mode='determinate',
            maximum=150
        )
        self.progress_bar.pack()
        
        self.progress_label = tk.Label(
            progress_frame,
            text="0/150 samples",
            font=('Arial', 12),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        self.progress_label.pack(pady=5)
        
        # Emotion breakdown
        self.breakdown_label = tk.Label(
            self.calib_frame,
            text="",
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='#cccccc',
            justify=tk.LEFT
        )
        self.breakdown_label.pack(pady=10)
        
        # Instructions
        self.instruction_label = tk.Label(
            self.calib_frame,
            text="Make the emotion shown above, then click 'Capture' when ready",
            font=('Arial', 12),
            bg='#2b2b2b',
            fg='#ffffff',
            wraplength=700
        )
        self.instruction_label.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(self.calib_frame, bg='#2b2b2b')
        button_frame.pack(pady=20)
        
        self.capture_button = tk.Button(
            button_frame,
            text="📸 Capture Sample",
            font=('Arial', 14, 'bold'),
            bg='#2196F3',
            fg='white',
            command=self.capture_sample,
            padx=30,
            pady=15,
            state=tk.DISABLED
        )
        self.capture_button.pack(side=tk.LEFT, padx=10)
        
        self.quit_button = tk.Button(
            button_frame,
            text="❌ Quit & Train",
            font=('Arial', 14, 'bold'),
            bg='#f44336',
            fg='white',
            command=self.quit_and_train,
            padx=30,
            pady=15,
            state=tk.DISABLED
        )
        self.quit_button.pack(side=tk.LEFT, padx=10)
    
    def start_camera_thread(self):
        """Start background thread for camera capture."""
        def camera_loop():
            while self.is_capturing:
                ret, frame = self.cap.read()
                if ret:
                    # Detect emotions
                    emo = self.fer.detect_emotions(frame)
                    self.latest_frame = frame
                    if len(emo) > 0:
                        self.latest_emotions = emo[0]['emotions']
        
        self.is_capturing = True
        self.camera_thread = threading.Thread(target=camera_loop, daemon=True)
        self.camera_thread.start()
    
    def update_video_feed(self):
        """Update the video feed display."""
        if self.latest_frame is not None:
            frame = self.latest_frame.copy()
            frame = cv2.resize(frame, (640, 480))
            
            # Draw text on frame
            if self.person_name and self.sample_count < 150:
                cv2.putText(frame, f"Sample: {self.sample_count}/150", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                current_emotion = self.emotion_sequence[self.sample_count] if self.sample_count < 150 else "DONE"
                cv2.putText(frame, f"Make: {current_emotion.upper()}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)
            
            self.video_label.config(image=photo)
            self.video_label.image = photo
        
        # Schedule next update
        self.root.after(30, self.update_video_feed)
    
    def start_calibration(self):
        """Start the calibration process."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter your name!")
            return
        
        self.person_name = name
        self.name_frame.pack_forget()
        self.calib_frame.pack(pady=10)
        
        self.capture_button.config(state=tk.NORMAL)
        self.quit_button.config(state=tk.NORMAL)
        
        self.update_display()
    
    def capture_sample(self):
        """Capture a training sample."""
        if self.latest_emotions is None:
            messagebox.showwarning("Warning", "No face detected! Please face the camera.")
            return
        
        if self.sample_count >= 150:
            messagebox.showinfo("Complete", "Already collected 150 samples!")
            return
        
        # Get current emotion
        current_emotion = self.emotion_sequence[self.sample_count]
        emotion_idx = self.calibration_manager.emotion_labels.index(current_emotion)
        
        # Add sample
        emotion_vector = [self.latest_emotions[label] for label in self.calibration_manager.emotion_labels]
        self.calibration_manager.add_training_sample(self.person_name, emotion_vector, emotion_idx)
        
        self.sample_count += 1
        self.update_display()
        
        # Check milestones
        if self.sample_count in [30, 60, 90, 120]:
            response = messagebox.askyesno(
                f"Milestone: {self.sample_count} samples",
                f"You've collected {self.sample_count}/150 samples!\n\nContinue calibration?"
            )
            if not response:
                self.quit_and_train()
                return
        
        # Auto-finish at 150
        if self.sample_count >= 150:
            messagebox.showinfo(
                "Complete!",
                "Collected all 150 samples!\n\nClick 'Quit & Train' to train your model."
            )
            self.capture_button.config(state=tk.DISABLED)
    
    def update_display(self):
        """Update the display with current progress."""
        if self.sample_count >= 150:
            self.emotion_label.config(text="✅ COMPLETE!")
            self.instruction_label.config(text="Click 'Quit & Train' to save your calibration model")
        else:
            current_emotion = self.emotion_sequence[self.sample_count]
            self.emotion_label.config(text=f"Make this emotion: {current_emotion.upper()}")
        
        self.progress_bar['value'] = self.sample_count
        self.progress_label.config(text=f"{self.sample_count}/150 samples")
        
        # Update emotion breakdown
        emotion_counts = self.calibration_manager.get_emotion_counts(self.person_name)
        breakdown_text = "Samples per emotion:\n"
        for emotion, count in emotion_counts.items():
            breakdown_text += f"  {emotion}: {count}  "
        self.breakdown_label.config(text=breakdown_text)
    
    def quit_and_train(self):
        """Quit and train the model."""
        if self.sample_count < 10:
            messagebox.showerror("Error", "Need at least 10 samples to train a model!")
            return
        
        response = messagebox.askyesno(
            "Train Model",
            f"Train model with {self.sample_count} samples?"
        )
        
        if response:
            # Train model
            success, message = self.calibration_manager.train_person_model(self.person_name)
            
            if success:
                messagebox.showinfo(
                    "Success!",
                    f"Model trained successfully!\n{message}\n\nSaved to person_models/"
                )
            else:
                messagebox.showerror("Error", f"Training failed: {message}")
            
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and close."""
        self.is_capturing = False
        self.cap.release()
        self.root.destroy()


# --- Main ---
root = tk.Tk()
app = CalibrationGUI(root, cap, fer)
root.protocol("WM_DELETE_WINDOW", app.cleanup)
root.mainloop()

print("\n✅ Calibration complete!")

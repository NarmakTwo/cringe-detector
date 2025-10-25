import json
import cv2
import numpy as np
import time
from fer import FER
import os
import tensorflow as tf
from tensorflow import keras
from collections import deque

print("Initializing FER...")

# Initialize FER
fer = FER()

# --- Face Tracking System ---
class FaceTracker:
    """
    Tracks faces across frames using position and size.
    Assigns consistent IDs to faces and manages person identification.
    """
    def __init__(self, max_distance=100, max_frames_missing=30):
        self.tracked_faces = {}  # {face_id: FaceInfo}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.person_names = {}  # {face_id: person_name}
        
    def _compute_distance(self, box1, box2):
        """Compute center distance between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections):
        """
        Update tracked faces with new detections.
        Returns: list of (face_id, box, emotions) tuples
        """
        current_frame_faces = set()
        matched_faces = []
        
        # Match detections to existing tracks
        unmatched_detections = []
        for detection in detections:
            box = detection['box']
            emotions = detection['emotions']
            
            best_match_id = None
            best_distance = float('inf')
            
            # Find best matching tracked face
            for face_id, face_info in self.tracked_faces.items():
                if face_info['frames_missing'] > 0:
                    continue
                    
                distance = self._compute_distance(box, face_info['last_box'])
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = face_id
            
            if best_match_id is not None:
                # Update existing track
                self.tracked_faces[best_match_id]['last_box'] = box
                self.tracked_faces[best_match_id]['frames_missing'] = 0
                self.tracked_faces[best_match_id]['position_history'].append(
                    (box[0] + box[2]/2, box[1] + box[3]/2)
                )
                current_frame_faces.add(best_match_id)
                matched_faces.append((best_match_id, box, emotions))
            else:
                unmatched_detections.append((box, emotions))
        
        # Create new tracks for unmatched detections
        for box, emotions in unmatched_detections:
            face_id = self.next_id
            self.next_id += 1
            self.tracked_faces[face_id] = {
                'last_box': box,
                'frames_missing': 0,
                'position_history': deque(maxlen=30),
                'first_seen': time.time()
            }
            current_frame_faces.add(face_id)
            matched_faces.append((face_id, box, emotions))
        
        # Update missing frames counter
        for face_id in list(self.tracked_faces.keys()):
            if face_id not in current_frame_faces:
                self.tracked_faces[face_id]['frames_missing'] += 1
                
                # Remove old tracks
                if self.tracked_faces[face_id]['frames_missing'] > self.max_frames_missing:
                    del self.tracked_faces[face_id]
                    if face_id in self.person_names:
                        del self.person_names[face_id]
        
        return matched_faces
    
    def assign_person(self, face_id, name):
        """Assign a person's name to a face ID."""
        self.person_names[face_id] = name
    
    def get_person_name(self, face_id):
        """Get the person's name for a face ID."""
        return self.person_names.get(face_id, f"Person {face_id}")
    
    def get_all_tracked_persons(self):
        """Get all currently tracked persons with their names."""
        result = []
        for face_id in self.tracked_faces.keys():
            if self.tracked_faces[face_id]['frames_missing'] == 0:
                result.append((face_id, self.get_person_name(face_id)))
        return result


# --- Per-Person Calibration System ---
class PersonCalibrationManager:
    """
    Manages individual calibration models for each person.
    Each person gets their own trained model.
    """
    def __init__(self):
        self.models = {}  # {person_name: keras.Model}
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.model_dir = "person_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_person_model(self, person_name):
        """Load a trained model for a specific person."""
        safe_name = self._safe_filename(person_name)
        model_path = os.path.join(self.model_dir, f"{safe_name}_model.keras")
        metadata_path = os.path.join(self.model_dir, f"{safe_name}_metadata.json")
        
        if not os.path.exists(model_path):
            return False
        
        self.models[person_name] = keras.models.load_model(model_path)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   {person_name}: {metadata['training_samples']} samples, {metadata['final_accuracy']:.2%} accuracy")
        
        return True
    
    def load_all_models(self):
        """Load all available person models."""
        if not os.path.exists(self.model_dir):
            return False
        
        loaded_count = 0
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_metadata.json'):
                with open(os.path.join(self.model_dir, filename), 'r') as f:
                    metadata = json.load(f)
                person_name = metadata['person_name']
                if self.load_person_model(person_name):
                    loaded_count += 1
        
        return loaded_count > 0
    
    def predict(self, person_name, emotion_vector):
        """Apply person-specific calibration, or use raw if no model available."""
        if person_name in self.models:
            # Use person-specific model
            emotion_array = np.array([emotion_vector], dtype=np.float32)
            predictions = self.models[person_name].predict(emotion_array, verbose=0)[0]
            return {label: float(pred) for label, pred in zip(self.emotion_labels, predictions)}
        else:
            # Fall back to raw emotions
            return {label: val for label, val in zip(self.emotion_labels, emotion_vector)}
    
    def has_model(self, person_name):
        """Check if a model exists for a person."""
        return person_name in self.models
    
    def _safe_filename(self, person_name):
        """Convert person name to safe filename."""
        safe = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
        return safe.replace(' ', '_').lower()


def load_calibration_models():
    """Load all trained person-specific calibration models."""
    calibration_manager = PersonCalibrationManager()
    
    print("Loading person-specific calibration models...")
    if not calibration_manager.load_all_models():
        print("❌ No calibration models found.")
        print("   Please run the calibration program first (calibrate.py)")
        return None
    
    print(f"✅ Loaded models for {len(calibration_manager.models)} person(s):")
    
    return calibration_manager


def process_video(video_path, calibration_manager, export_mode=False):
    """
    Process a video and detect who cringes last (or never cringes).
    If export_mode is True, creates an annotated output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    tracker = FaceTracker()
    frame_count = 0
    pending_identification = set()
    cringe_events = []  # Track all cringe events: [(person, timestamp, frame, level)]
    people_who_cringed = set()  # Track which people have cringed
    all_people_seen = set()  # Track all people in the video
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if exporting
    video_writer = None
    if export_mode:
        output_path = video_path.rsplit('.', 1)[0] + '_annotated.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        print(f"\n📹 Exporting annotated video to: {output_path}")
    
    print(f"\n🎥 Processing video: {video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    print(f"   Goal: Find who cringes LAST (or never cringes)!")
    if not export_mode:
        print("\n   Press 'q' to quit early")
        print("   Press 'i' to identify a person\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n📹 Reached end of video")
            break
        
        frame_count += 1
        original_frame = frame.copy()
        frame = cv2.resize(frame, (640, 480))
        
        # Process every 3rd frame for detection
        process_this_frame = (frame_count % 3 == 0)
        
        if process_this_frame:
            emo_detections = fer.detect_emotions(frame)
            
            # Convert to format for tracker
            detections = [{'box': e['box'], 'emotions': e['emotions']} for e in emo_detections]
            tracked_faces = tracker.update(detections)
        else:
            tracked_faces = []
        
        # Create display frame with annotations
        display_frame = frame.copy()
        
        # Add timestamp overlay
        timestamp = frame_count / fps
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        time_text = f"{minutes:02d}:{seconds:05.2f}"
        
        # Draw timestamp background
        cv2.rectangle(display_frame, (10, 10), (180, 50), (0, 0, 0), -1)
        cv2.putText(display_frame, time_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if process_this_frame and len(tracked_faces) > 0:
            for face_id, box, raw_emotions in tracked_faces:
                x, y, w, h = box
                person_name = tracker.get_person_name(face_id)
                
                # Track all people seen
                if face_id in tracker.person_names:
                    all_people_seen.add(person_name)
                
                # Apply person-specific calibration
                emotion_vector = [raw_emotions[label] for label in calibration_manager.emotion_labels]
                corrected_emotions = calibration_manager.predict(person_name, emotion_vector)
                
                # Get top 3 emotions
                sorted_emotions = sorted(corrected_emotions.items(), key=lambda x: x[1], reverse=True)
                top_3_emotions = sorted_emotions[:3]
                major_emotion = top_3_emotions[0][0]
                
                # Determine color based on model availability
                has_model = calibration_manager.has_model(person_name)
                if face_id in tracker.person_names:
                    color = (0, 255, 0) if has_model else (0, 165, 255)  # Green if model, Orange if no model
                else:
                    color = (0, 0, 255)  # Red if unknown
                
                # Draw face rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Create info box background
                info_box_height = 90
                info_y_start = max(y - info_box_height - 5, 0)
                
                # Semi-transparent background for info
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x, info_y_start), (x + w, y - 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Draw person name with model indicator
                model_indicator = "✓" if has_model else "○"
                name_text = f"{person_name}{model_indicator}"
                cv2.putText(display_frame, name_text, (x + 5, info_y_start + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw top 3 emotions with percentages
                for i, (emotion, confidence) in enumerate(top_3_emotions):
                    emotion_text = f"{emotion}: {confidence*100:.1f}%"
                    y_pos = info_y_start + 40 + (i * 15)
                    
                    # Highlight top emotion
                    emotion_color = (0, 255, 255) if i == 0 else (200, 200, 200)
                    cv2.putText(display_frame, emotion_text, (x + 5, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, emotion_color, 1)
                
                # Check if needs identification
                if face_id not in tracker.person_names and face_id not in pending_identification:
                    pending_identification.add(face_id)
                
                # Track disgust detection
                if major_emotion == "disgust" and corrected_emotions[major_emotion] > 0.70:
                    # Record cringe event
                    cringe_events.append((
                        person_name,
                        timestamp,
                        frame_count,
                        corrected_emotions[major_emotion]
                    ))
                    people_who_cringed.add(person_name)
                    
                    if not export_mode:
                        print(f"😬 {person_name} cringed at {timestamp:.2f}s ({corrected_emotions[major_emotion]:.1%})")
        
        # Progress indicator
        progress = (frame_count / total_frames) * 100
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write to output video if exporting
        if video_writer:
            video_writer.write(display_frame)
            
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
        else:
            cv2.imshow('Video Processing', display_frame)
        
        # Handle keyboard input (only in non-export mode)
        if not export_mode:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('i'):
                # Manual identification
                persons = tracker.get_all_tracked_persons()
                if persons:
                    print("\nCurrently tracked persons:")
                    for i, (face_id, person_name) in enumerate(persons):
                        has_model = calibration_manager.has_model(person_name)
                        model_status = "✓ calibrated" if has_model else "○ not calibrated"
                        print(f"  {i+1}. {person_name} ({model_status})")
                    try:
                        idx = int(input("Enter person # to rename: ").strip()) - 1
                        if 0 <= idx < len(persons):
                            face_id = persons[idx][0]
                            new_name = input("Enter new name: ").strip()
                            if new_name:
                                tracker.assign_person(face_id, new_name)
                                print(f"✓ Renamed to: {new_name}")
                    except (ValueError, IndexError):
                        print("Invalid selection")
            
            # Handle pending identifications
            if pending_identification and frame_count % 60 == 0:
                face_id = pending_identification.pop()
                print(f"\n🆕 New face detected (ID: {face_id})")
                person_name = input("Enter person's name (or press ENTER to skip): ").strip()
                if person_name:
                    tracker.assign_person(face_id, person_name)
                    print(f"✓ Registered as: {person_name}")
                    if not calibration_manager.has_model(person_name):
                        print(f"   ⚠️  No calibration model for {person_name}, using raw FER")
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\n\n✅ Annotated video saved to: {output_path}")
    cv2.destroyAllWindows()
    
    # Determine winner: person who cringed last, or never cringed
    winner = None
    winner_timestamp = None
    winner_frame = None
    
    # Find people who never cringed
    people_who_didnt_cringe = all_people_seen - people_who_cringed
    
    if people_who_didnt_cringe:
        # Winner is someone who never cringed
        winner = list(people_who_didnt_cringe)[0] if len(people_who_didnt_cringe) == 1 else "Multiple people"
        winner_timestamp = total_frames / fps  # Made it to the end
        winner_frame = total_frames
    elif cringe_events:
        # Winner is the last person to cringe
        last_cringe = cringe_events[-1]
        winner = last_cringe[0]
        winner_timestamp = last_cringe[1]
        winner_frame = last_cringe[2]
    
    return {
        'winner': winner,
        'timestamp': winner_timestamp,
        'frame': winner_frame,
        'all_events': cringe_events,
        'all_people': all_people_seen,
        'people_who_cringed': people_who_cringed,
        'people_who_didnt_cringe': people_who_didnt_cringe
    }


# --- Main Program ---
print("\n" + "="*50)
print("CRINGE DETECTOR - Video Processor")
print("="*50)

# Mode selection
print("\nSelect mode:")
print("1. Use calibration models (personalized detection)")
print("2. Use base FER (no calibration)")
print("3. Quit")

mode_choice = input("\nEnter choice: ").strip()

if mode_choice == '3':
    print("Exiting...")
    exit(0)

if mode_choice == '2':
    # Use base FER without calibration
    print("\n⚠️  Using base FER without calibration")
    print("   Detection will use raw FER predictions for all faces\n")
    
    # Create a dummy calibration manager that always returns raw emotions
    class BaseFERManager:
        def __init__(self):
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.models = {}
        
        def predict(self, person_name, emotion_vector):
            """Return raw emotions without calibration."""
            return {label: val for label, val in zip(self.emotion_labels, emotion_vector)}
        
        def has_model(self, person_name):
            """Always return False - no models available."""
            return False
    
    calibration_manager = BaseFERManager()
    
elif mode_choice == '1':
    # Load calibration models
    calibration_manager = load_calibration_models()
    if calibration_manager is None:
        print("\n❌ No calibration models found.")
        print("   Please run 'calibrate.py' first or use mode 2 (base FER)")
        exit(1)
else:
    print("Invalid choice. Exiting...")
    exit(1)

# Get video path
video_path = input("\nEnter video path to process: ").strip()
if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    exit(1)

# Process video
result = process_video(video_path, calibration_manager)

# After processing, show results
if result:
    print(f"\n" + "="*60)
    print(f"🎯 CRINGE CHALLENGE RESULTS")
    print(f"="*60)
    
    print(f"\n👥 Participants: {', '.join(result['all_people'])}")
    
    if result['people_who_didnt_cringe']:
        print(f"\n🏆 WINNER (Never Cringed): {', '.join(result['people_who_didnt_cringe'])}")
        print(f"   They made it through the entire video without cringing!")
    elif result['winner']:
        print(f"\n🏆 WINNER (Last to Cringe): {result['winner']}")
        minutes = int(result['timestamp'] // 60)
        seconds = result['timestamp'] % 60
        print(f"   Held out until: {minutes:02d}:{seconds:05.2f}")
        print(f"   Frame: {result['frame']}")
    else:
        print(f"\n🤷 No one detected in the video!")
    
    if result['all_events']:
        print(f"\n📊 Cringe Timeline ({len(result['all_events'])} events):")
        print(f"="*60)
        
        # Sort events by timestamp
        sorted_events = sorted(result['all_events'], key=lambda x: x[1])
        
        for i, (person, timestamp, frame, level) in enumerate(sorted_events, 1):
            minutes = int(timestamp // 60)
            seconds = timestamp % 60
            print(f"   {i}. {person} at {minutes:02d}:{seconds:05.2f} ({level:.1%} disgust)")
        
        # Show who cringed vs who didn't
        print(f"\n📈 Summary:")
        print(f"   Cringed: {', '.join(result['people_who_cringed']) if result['people_who_cringed'] else 'No one'}")
        print(f"   Didn't Cringe: {', '.join(result['people_who_didnt_cringe']) if result['people_who_didnt_cringe'] else 'No one'}")
    else:
        print(f"\n✨ No cringe detected from anyone! Everyone is a winner!")
    
    print(f"="*60)
    
    export_choice = input("\n📹 Export annotated video? (y/n): ").strip().lower()
    if export_choice == 'y':
        print("\n🎬 Re-processing video to create annotated export...")
        process_video(video_path, calibration_manager, export_mode=True)

print("\n✅ Done!")

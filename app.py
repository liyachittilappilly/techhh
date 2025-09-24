import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Threshold distance for hands being "close" (in pixels)
DISTANCE_THRESHOLD = 150

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and check for hand proximity
    if results.multi_hand_landmarks:
        # Get landmarks for both hands
        hand_landmarks = results.multi_hand_landmarks
        
        # Draw landmarks on frame
        for hand_landmark in hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
        
        # Check if two hands are detected
        if len(hand_landmarks) == 2:
            # Get wrist positions (landmark 0)
            hand1_wrist = hand_landmarks[0].landmark[0]
            hand2_wrist = hand_landmarks[1].landmark[0]
            
            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x1, y1 = int(hand1_wrist.x * w), int(hand1_wrist.y * h)
            x2, y2 = int(hand2_wrist.x * w), int(hand2_wrist.y * h)
            
            # Calculate distance between wrists
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Draw connection line between wrists
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Check if hands are close enough
            if distance < DISTANCE_THRESHOLD:
                # Display "Holding Hands" message
                cv2.putText(frame, "Holding Hands!", (w//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Draw circles at wrist positions
                cv2.circle(frame, (x1, y1), 15, (255, 0, 0), -1)
                cv2.circle(frame, (x2, y2), 15, (255, 0, 0), -1)
    
    # Display output frame
    cv2.imshow('Hand Detection', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
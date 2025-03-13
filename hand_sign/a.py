# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

model = load_model('hand_gesture_model_improved.h5')
print("Model loaded successfully.")


# Function to predict a new image
def predict_image(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))  # Resize the image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image (rescale pixel values to [0, 1])

    # Make the prediction
    prediction = model.predict(img_array)

    # Define class names (0-9 and A-Z)
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    print(f"The image is predicted to be class {predicted_class} with a confidence of {confidence:.2f}")


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        print("Hand detected!")
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Add padding around the hand
            padding = 50
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop the hand region
            hand_image = frame[y_min:y_max, x_min:x_max]

            # Save the cropped hand image
            hand_image_path = 'hand_image1.jpg'
            cv2.imwrite(hand_image_path, hand_image)
            print("Hand image saved as 'hand_image1.jpg'")

            # Display the cropped hand image
            cv2.imshow('Cropped Hand', hand_image)

            # Predict the gesture using the saved model
            predict_image(model, hand_image_path)
    else:
        print("No hand detected.")

    # Show the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

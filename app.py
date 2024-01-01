import cv2 as cv
import mediapipe as mp
import time

# Open the default camera (camera index 0)
video = cv.VideoCapture(0)

# Initialize the Mediapipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDrawing = mp.solutions.drawing_utils 

# frame rate
pTime = 0
cTime = 0


while True:
    # Read a frame from the video capture
    success, img = video.read()

    # Convert the BGR image to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Process the frame using the Mediapipe Hands module
    results = hands.process(imgRGB)
    

    # Print the landmarks of detected hands (if any)
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)
            for id ,lm in enumerate(handLandmarks.landmark):
                print(f'id : {id}  lm :  {lm}') 

            # Draw lines connecting specific hand landmarks
            landmarks_to_connect = [mpHands.HandLandmark.WRIST, mpHands.HandLandmark.THUMB_TIP,
                                     mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_TIP,
                                     mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.PINKY_TIP]

            for landmark in landmarks_to_connect:
                landmark_point = handLandmarks.landmark[landmark]
                cx, cy = int(landmark_point.x * img.shape[1]), int(landmark_point.y * img.shape[0])
                cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)


            # Draw lines connecting the specified landmarks
            connections = [(mpHands.HandLandmark.WRIST, mpHands.HandLandmark.THUMB_TIP),
                           (mpHands.HandLandmark.THUMB_TIP, mpHands.HandLandmark.INDEX_FINGER_TIP),
                           (mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_TIP),
                           (mpHands.HandLandmark.MIDDLE_FINGER_TIP, mpHands.HandLandmark.RING_FINGER_TIP),
                           (mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.PINKY_TIP)]

            for connection in connections:
                start_point = handLandmarks.landmark[connection[0]]
                end_point = handLandmarks.landmark[connection[1]]
                start_x, start_y = int(start_point.x * img.shape[1]), int(start_point.y * img.shape[0])
                end_x, end_y = int(end_point.x * img.shape[1]), int(end_point.y * img.shape[0])
                cv.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
            
    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,50,255), 3)
    # Display the original frame
    cv.imshow('Video', img)

    # Break the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()

# Close all OpenCV windows
cv.destroyAllWindows()

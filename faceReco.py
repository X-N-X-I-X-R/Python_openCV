import cv2

def detect_faces():
    # טען את המכשיר המצלמה
    cap = cv2.VideoCapture(0)

    # טען את מכשיר זיהוי הפנים של OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # קרא פריימים מהמצלמה
        ret, frame = cap.read()

        # המרת התמונה לגווני האפור (gray scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # זיהוי פנים בתמונה
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # סימון פנים בתמונה
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # הצגת התמונה עם המסגרת
        cv2.imshow('Face Detection', frame)

        # יציאה מהלולאה כאשר לוחץ על המקש 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # שחרור משאבים
    cap.release()


# קריאה לפונקציה







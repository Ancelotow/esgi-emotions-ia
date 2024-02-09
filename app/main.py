import time
import cv2
from api.api import feeling_detection


def feeling_detector(x, y, w, h):
    roi_gray = gray[y:y + h, x:x + w]
    roi_resized = cv2.resize(roi_gray, (48, 48))  # Resize the ROI to 48x48 and convert to grayscale
    roi_flattened = roi_resized.flatten().reshape(1, -1)  # Flatten and reshape the image
    return feeling_detection(roi_flattened)


def face_detector(img, gray):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(48, 48)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        feeling = feeling_detector(x, y, w, h)
        cv2.putText(img, feeling, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


def eyes_detector(img, gray):
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]


if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier('lib/haarcascade.xml')
    eyeCascade = cv2.CascadeClassifier('lib/eye_cascade.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while (True):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_detector(img, gray)
        eyes_detector(img, gray)

        cv2.imshow('video', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
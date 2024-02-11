import tkinter as tk
from PIL import Image, ImageTk
import cv2
from api.api import feeling_detection, age_detection

faceCascade = cv2.CascadeClassifier('lib/haarcascade.xml')
eyeCascade = cv2.CascadeClassifier('lib/eye_cascade.xml')
cap = cv2.VideoCapture(0)
text_id = None


def feeling_detector(x, y, w, h, gray):
    roi_gray = gray[y:y + h, x:x + w]
    return feeling_detection(roi_gray)


def age_detector(x, y, w, h, gray):
    roi_gray = gray[y:y + h, x:x + w]
    return age_detection(roi_gray)


def face_detector(img, gray):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(48, 48)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        feeling = feeling_detector(x, y, w, h, gray)
        age = age_detector(x, y, w, h, gray)

        # Create a semi-transparent rectangle as a background for the text
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y - 80), (x + w, y), (0, 0, 0), -1)

        # Apply the overlay
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Put the text on the image
        cv2.putText(img, f"Emotion : {feeling}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, f"Age : {age}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def eyes_detector(img, gray):
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in eyes:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]


def update_frame():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if img is None:
        print("L'image n'a pas pu être chargée.")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_detector(img, gray)
        eyes_detector(img, gray)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image)
        video_label.config(image=photo)
        video_label.image = photo
    # Update the text in the canvas
    window.after(20, update_frame)


def update_video_size(event):
    window_width = window.winfo_width()  # Get the current window width
    window_height = window.winfo_height()  # Get the current window height

    cap.set(3, window_width)  # Set the video width to the window width
    cap.set(4, window_height)  # Set the video height to the window height


if __name__ == '__main__':
    window = tk.Tk()
    video_label = tk.Label(window)
    video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Make the video label fill the window
    window.bind('<Configure>', update_video_size)  # Update the video size when the window size changes
    update_frame()
    window.mainloop()
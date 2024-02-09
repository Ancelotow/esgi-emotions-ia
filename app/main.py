import tkinter as tk
from PIL import Image, ImageTk
import cv2
from api.api import feeling_detection

faceCascade = cv2.CascadeClassifier('lib/haarcascade.xml')
eyeCascade = cv2.CascadeClassifier('lib/eye_cascade.xml')
cap = cv2.VideoCapture(0)
text_id = None

def feeling_detector(x, y, w, h, gray):
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
        feeling = feeling_detector(x, y, w, h, gray)
        canvas.itemconfig(text_id, text=feeling)
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


def update_frame():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
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


if __name__ == '__main__':
    window = tk.Tk()
    video_label = tk.Label(window)
    video_label.pack()

    cap.set(3, 640)
    cap.set(4, 480)

    # Create a canvas
    canvas = tk.Canvas(window, width=200, height=200)
    canvas.pack()
    canvas.create_rectangle(50, 50, 200, 200, fill="white")

    # Add Text
    text_id = canvas.create_text(125, 125, text="Unknown", fill="black")

    # Add Image (with resize)
    img = Image.open("assets/ic_feeling.png")
    img = img.resize((20, 20))
    photoImg = ImageTk.PhotoImage(img)
    canvas.create_image(30, 30, image=photoImg)

    update_frame()
    window.mainloop()
import cv2
import os
import numpy as np
from PIL import Image
import tkinter as tk

# Create the window
window = tk.Tk()

# Global notification label for feedback
Notification = tk.Label(window, text="", bg="white", font=('times', 18, 'bold'))

# Initialize the recognizer and detector globally
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def trainimg():
    try:
        faces, Ids = getImagesAndLabels("TrainingImage")  # Get faces and IDs from the folder
        if not faces or not Ids:
            raise ValueError("No faces found in the training images.")

        recognizer.train(faces, np.array(Ids))  # Train the model

        # Ensure the folder exists for saving the model
        if not os.path.exists('TrainingImageLabel'):
            os.makedirs('TrainingImageLabel')

        try:
            recognizer.save(r"TrainingImageLabel\Trainner.yml")  # Save the trained model
            res = "Model Trained Successfully"
            Notification.configure(text=res, bg="olive drab", width=50)
        except Exception as e:
            res = f'Error saving model: {str(e)}'
            Notification.configure(text=res, bg="SpringGreen3", width=50)

    except Exception as e:
        res = f"Error: {str(e)}"
        Notification.configure(text=res, bg="SpringGreen3", width=50)

    Notification.place(x=250, y=400)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    faceSamples = []
    Ids = []

    # Looping through all the image paths and loading the Ids and images
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # Convert image to grayscale
        imageNp = np.array(pilImage, 'uint8')

        # Extract ID from the image filename (format: vish.1.1.jpg)
        try:
            filename = os.path.split(imagePath)[-1]
            # Remove leading/trailing spaces if any
            filename = filename.strip()

            # Extract the second-to-last part (ID) from the filename
            Id = int(filename.split('.')[-2])  # Extract ID from the second-to-last part
        except ValueError:
            print(f"Skipping invalid file: {imagePath}")
            continue
        
        # Detect faces in the image
        faces = detector.detectMultiScale(imageNp)
        
        # If faces are detected, add them to faceSamples and add the ID to Ids
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)

    return faceSamples, Ids


# Configure window to handle closing event
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

import cv2
import numpy as np
from tkinter import filedialog, Tk, Label, Button, Frame
from PIL import Image, ImageTk

# Model file paths
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACE_PROTO = "deploy.prototxt"

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MEAN_VALUES = (104.0, 117.0, 123.0)

# Load models
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

# Predict age & gender
def detect_age_gender(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), MEAN_VALUES, swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN_VALUES, swapRB=False)

            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            label = f"{gender}, {age}"

            # Draw pastel box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (147, 112, 219), 2)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 10, 5)
            lx1 = x1
            ly1 = y1 - 30 if y1 - 30 > 10 else y1 + 30
            lx2 = x1 + label_size[0] + 10
            ly2 = ly1 + label_size[1] + 10

            cv2.rectangle(image, (lx1, ly1), (lx2, ly2), (255, 228, 250), -1)
            cv2.putText(image, label, (lx1 + 5, ly1 + label_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (147, 112, 219), 5)

    return image

# GUI class
class AgeGenderApp:
    def __init__(self, master):
        self.master = master
        master.title("🌸 Pastel Age & Gender Detector")
        master.configure(bg="#fffafc")

        self.label = Label(master, text="Upload a photo to bloom predictions 🌷",
                           bg="#fffafc", fg="#d17ea5", font=("Comic Sans MS", 13, "bold"))
        self.label.pack(pady=12)

        self.upload_button = Button(master, text="🖼️ Choose Image", command=self.upload_image,
                                    bg="#ffb6c1", fg="white", font=("Comic Sans MS", 11, "bold"),
                                    padx=12, pady=5, relief="flat")
        self.upload_button.pack(pady=10)

        self.frame = Frame(master, bg="#ffeef4", bd=3, relief="ridge")
        self.frame.pack(pady=10)

        self.image_label = Label(self.frame, bg="#ffeef4")
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            output = detect_age_gender(file_path)
            cv2.imwrite("output_pastel.jpg", output)

            # Resize for display (compact but readable)
            img = Image.open("output_pastel.jpg")
            display_width = 400
            if img.width > display_width:
                ratio = display_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((display_width, new_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.master.geometry(f"{img.width + 50}x{img.height + 150}")

# Launch GUI
if __name__ == "__main__":
    root = Tk()
    app = AgeGenderApp(root)
    root.mainloop()

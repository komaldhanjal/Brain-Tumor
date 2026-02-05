import os
import tkinter as tk
from tkinter import Frame, Label, Button
from PIL import Image, ImageTk
import numpy as np
import joblib

from database import MySQLDB   # 🔥 DATABASE IMPORT

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR  = os.path.join(BASE_DIR, "Testing")

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

IMG_SIZE = (460, 460)
PREPROCESS_SIZE = (200, 200)

#LOAD MODEL
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ================= DB CONNECTION =================
db = MySQLDB(
    host="localhost",
    user="root",
    password="051102",
    database="brain_tumor_ai"
)

# ================= APP =================
app = tk.Tk()
app.title("Brain Tumor Detection")
app.geometry("1200x750")
app.configure(bg="#2b2f4b")

images = []
img_index = 0

# ================= FUNCTIONS =================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize(PREPROCESS_SIZE)
    img = np.array(img).flatten().reshape(1, -1)
    img = scaler.transform(img)
    return img

def predict_current_image(img_path):
    data = preprocess_image(img_path)
    pred = model.predict(data)[0]

    image_name = os.path.basename(img_path)

    if pred == 1:
        result = "Tumor Detected"
        result_label.config(text="🧠 Tumor Detected", fg="#ef4444")
    else:
        result = "No Tumor"
        result_label.config(text="✅ No Tumor", fg="#22c55e")

    # 🔥 SAVE TO DATABASE
    try:
        db.insert_prediction(
            image_name=image_name,
            prediction=result,
            model_output=int(pred)
        )
    except Exception as e:
        print("Database Error:", e)

def load_images(folder):
    global images, img_index
    images = []
    img_index = 0

    if not os.path.exists(folder):
        result_label.config(text="Folder not found", fg="white")
        return

    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            images.append(os.path.join(folder, f))

    if images:
        show_image()

def show_image():
    img_path = images[img_index]
    img = Image.open(img_path).resize(IMG_SIZE)
    img = ImageTk.PhotoImage(img)

    img_label.image = img
    img_label.config(image=img)

    counter_label.config(
        text=f"Image {img_index+1} / {len(images)}",
        fg="white"
    )

    predict_current_image(img_path)

def next_img():
    global img_index
    if img_index < len(images) - 1:
        img_index += 1
        show_image()

def prev_img():
    global img_index
    if img_index > 0:
        img_index -= 1
        show_image()

def clear_screen():
    global images, img_index
    images = []
    img_index = 0
    img_label.config(image="")
    result_label.config(text="Screen Cleared", fg="white")
    counter_label.config(text="")

# ================= UI =================
Label(
    app,
    text="🧠 Brain Tumor Detection System",
    font=("Segoe UI", 26, "bold"),
    fg="#93c5fd",
    bg="#2b2f4b"
).pack(pady=10)

main = Frame(app, bg="#2b2f4b")
main.pack(fill="both", expand=True)

# LEFT PANEL
left = Frame(main, bg="#1f2333", width=260)
left.pack(side="left", fill="y", padx=10, pady=10)

Label(left, text="TRAIN", fg="white", bg="#1f2333",
      font=("Segoe UI", 15, "bold")).pack(pady=8)

for cls in ["yes", "pituitary_tumor", "no_tumor"]:
    Button(
        left, text=cls.upper(),
        width=20, height=2,
        bg="#2563eb", fg="white",
        font=("Segoe UI", 11, "bold"),
        relief="flat",
        command=lambda c=cls: load_images(os.path.join(TRAIN_DIR, c))
    ).pack(pady=5)

Label(left, text="TEST", fg="white", bg="#1f2333",
      font=("Segoe UI", 15, "bold")).pack(pady=15)

for cls in ["pituitary_tumor", "meningioma_tumor", "glioma_tumor", "no_tumor"]:
    Button(
        left, text=cls.upper(),
        width=20, height=2,
        bg="#2563eb", fg="white",
        font=("Segoe UI", 11, "bold"),
        relief="flat",
        command=lambda c=cls: load_images(os.path.join(TEST_DIR, c))
    ).pack(pady=5)

# CENTER
center = Frame(main, bg="#2b2f4b")
center.pack(side="left", fill="both", expand=True)

img_label = Label(center, bg="#2b2f4b")
img_label.pack(pady=15)

result_label = Label(
    center,
    text="Select Dataset",
    bg="#2b2f4b",
    font=("Segoe UI", 18, "bold")
)
result_label.pack(pady=5)

counter_label = Label(
    center,
    bg="#2b2f4b",
    font=("Segoe UI", 12, "bold")
)
counter_label.pack(pady=5)

nav = Frame(center, bg="#2b2f4b")
nav.pack(pady=10)

Button(nav, text="◀ Prev", width=12, bg="#2563eb",
       fg="white", font=("Segoe UI", 11, "bold"),
       command=prev_img).pack(side="left", padx=10)

Button(nav, text="Next ▶", width=12, bg="#2563eb",
       fg="white", font=("Segoe UI", 11, "bold"),
       command=next_img).pack(side="left", padx=10)

Button(nav, text="🧹 Clear", width=12, bg="#dc2626",
       fg="white", font=("Segoe UI", 11, "bold"),
       command=clear_screen).pack(side="left", padx=10)

app.mainloop()

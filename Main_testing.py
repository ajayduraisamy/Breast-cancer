# Main_testing.py
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox

import os
import cv2
import numpy as np

from PIL import Image, ImageTk

# --- Keras / TF ---
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# (Optional) if you use it elsewhere
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------- Utilities --------------

def safe_get_class_names(dataset_dir="./dataset"):
    """
    Returns class names by reading immediate subfolders of dataset_dir.
    Robust across OS path formats.
    """
    classes = []
    if not os.path.isdir(dataset_dir):
        return classes
    for entry in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, entry)
        if os.path.isdir(full):
            classes.append(entry)
    return classes

def path_to_tensor(img_path, width=224, height=224):
    """
    Loads an image, resizes, converts to array and adds a batch dimension.
    Uses TensorFlow's utils (compatible with TF 2.x).
    """
    img = load_img(img_path, target_size=(width, height))
    x = img_to_array(img)  # (H, W, 3)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)
    return x

# -------------- Tkinter Windows --------------

class Window(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        # App constants
        self.bg_image_path = "nerve.jpg"   # keep your file
        self.logo_path = "logo.jfif"       # keep your file
        self.model_path = "trained_model_DNN1.h5"  # your trained model
        self.input_size = (224, 224)       # change to what your model expects

        # Load model once
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model:\n{e}")
            self.model = None

        # Get class names once
        self.class_names = safe_get_class_names("./dataset")
        if not self.class_names:
            # Fallback if dataset folders are not present in deployment
            self.class_names = ["Class_0", "Class_1"]

        # Background
        try:
            bg_img = Image.open(self.bg_image_path).resize((1400, 1400))
            self.background_image = ImageTk.PhotoImage(bg_img)
            self.background_label = Label(self, image=self.background_image)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            # If background not available, just skip without crashing
            pass

        self.master.title("Breast Cancer Detection")

        title = tk.Label(
            self.master,
            text="Breast Cancer Detection",
            fg="Black",
            bg="white",
            font="Helvetica 20 bold italic"
        )
        title.pack()
        title.place(x=550, y=0)

        self.browse_btn = Button(
            self,
            command=self.query,
            text="Browse Input",
            fg="blue",
            activebackground="dark red",
            width=20
        )
        self.browse_btn.place(x=550, y=475)

        # Placeholder image (logo)
        try:
            logo_img = Image.open(self.logo_path).resize((250, 250))
            self.logo_render = ImageTk.PhotoImage(logo_img)
            self.image_preview = Label(
                self,
                image=self.logo_render,
                borderwidth=15,
                highlightthickness=5,
                height=224,
                width=224,
                bg='white'
            )
            self.image_preview.image = self.logo_render
            self.image_preview.place(x=500, y=200)
        except Exception:
            # if logo missing, create an empty placeholder
            self.image_preview = Label(
                self,
                text="No Preview",
                borderwidth=15,
                highlightthickness=5,
                height=15,
                width=30,
                bg='white'
            )
            self.image_preview.place(x=500, y=200)

        # Result text box
        self.result_box = Text(self, height=5, width=50)
        self.result_box.place(x=460, y=550)

    def query(self, event=None):
        # Pick a single image file
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )

        if not file_path:
            messagebox.showerror("Error", "No file selected")
            return

        # Preview
        try:
            preview = Image.open(file_path).resize((250, 250))
            preview_render = ImageTk.PhotoImage(preview)
            self.image_preview.configure(image=preview_render)
            self.image_preview.image = preview_render
        except Exception as e:
            messagebox.showerror("Error", f"Unable to preview image:\n{e}")
            return

        if self.model is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return

        # Preprocess & predict
        try:
            tensor = path_to_tensor(file_path, width=self.input_size[0], height=self.input_size[1]).astype(np.float32)
            tensor /= 255.0  # normalize 0-1 (adjust if your training used another scheme)

            preds = self.model.predict(tensor)
            # Handle different model output shapes
            if preds.ndim == 2 and preds.shape[1] > 1:
                # multi-class softmax
                idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds, axis=1)[0])
            else:
                # binary (sigmoid) -> convert to two-class style
                p1 = float(preds[0][0]) if preds.ndim == 2 else float(preds[0])
                idx = int(p1 >= 0.5)
                confidence = p1 if idx == 1 else (1.0 - p1)

            label = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"

            # Show result
            self.result_box.delete("1.0", END)
            self.result_box.insert(END, f"Predicted: {label}\nConfidence: {confidence:.4f}")

            # Optional popup
            # messagebox.showinfo("Result", f"{label} ({confidence:.2%})")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict:\n{e}")

class LoginWindow(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.config(bg='white')
        self.master.title("Login")

        # Background (optional)
        try:
            bg_image = Image.open("img.jpg").resize((1400, 1400))
            bg_render = ImageTk.PhotoImage(bg_image)
            self.background_label = Label(self, image=bg_render)
            self.background_label.image = bg_render
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            pass

        self.pack(fill=BOTH, expand=1)

        w = Label(self, text="Login Page", fg="#e6f2ff", bg="black", font="Helvetica 20 bold italic")
        w.pack()
        w.place(x=650, y=50)

        self.username_label = Label(self, text="Username:")
        self.username_label.place(x=550, y=100)

        self.username_entry = Entry(self)
        self.username_entry.place(x=650, y=100)

        self.password_label = Label(self, text="Password:")
        self.password_label.place(x=550, y=150)

        self.password_entry = Entry(self, show="*")
        self.password_entry.place(x=650, y=150)

        self.login_button = Button(self, text="Login", command=self.login)
        self.login_button.place(x=650, y=200)

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        # Simple demo auth
        valid = {
            "user": "123",
            "Aj": "123",
            "admin": "admin"
        }

        if username in valid and valid[username] == password:
            self.master.switch_frame(Window)
        else:
            messagebox.showerror("Error", "Invalid username or password")

class MainApplication(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("skin prick detection")
        self.geometry("1400x720")
        self.current_frame = None
        self.switch_frame(LoginWindow)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack()

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()

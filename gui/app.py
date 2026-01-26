import customtkinter as ctk
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import core.train as train
import utils.utils as utils

scriptpath = "../"
sys.path.append(os.path.abspath(scriptpath))
import core.train as train

ctk.set_appearance_mode("Dark")  # Dark / Light
ctk.set_default_color_theme("green")  # Thème vert

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.running = False
        self.loss = []

        self.title("GUI Inversion")
        self.geometry("1500x900")

        self.image = None
        self.photo = None
        self.device = "cuda"
        self.model = None

        self.build_ui()
        self.alpha = {16:1.0, 64:0.0, 512:0.0}

    def build_ui(self):        
        # ==== COLONNE BOUTONS ====
        btn_frame = ctk.CTkFrame(self, width=120, height=300, fg_color="transparent")
        btn_frame.place(x=40, y=40)

        ctk.CTkButton(btn_frame, text="Load", command=self.load_image, width=100).pack(pady=20)

        # ==== SLIDERS ALPHA ====
        self.slider1 = ctk.CTkSlider(
            btn_frame, from_=0, to=1, command=self.update_alpha1
        )
        self.slider1.pack(pady=(20, 5))
        self.slider1.set(1)

        self.label_alpha1 = ctk.CTkLabel(btn_frame, text="α1 = 1")
        self.label_alpha1.pack()

        self.slider2 = ctk.CTkSlider(
            btn_frame, from_=0, to=1, command=self.update_alpha2
        )
        self.slider2.pack(pady=(20, 5))
        self.slider2.set(0)

        self.label_alpha2 = ctk.CTkLabel(btn_frame, text="α2 = 0")
        self.label_alpha2.pack()

        self.slider3 = ctk.CTkSlider(
            btn_frame, from_=0, to=1, command=self.update_alpha3
        )
        self.slider3.pack(pady=(20, 5))
        self.slider3.set(0)

        self.label_alpha3 = ctk.CTkLabel(btn_frame, text="α3 = 0")
        self.label_alpha3.pack()

        ctk.CTkButton(btn_frame, text="Inv", command=self.inv_action, width=100).pack(pady=20)
        ctk.CTkButton(btn_frame, text="Qual", command=self.qual_action, width=100).pack(pady=20)
        ctk.CTkButton(btn_frame, text="Resume", command=self.resume, width=100).pack(pady=20)
        ctk.CTkButton(btn_frame, text="Stop", command=self.stop_action, width=100).pack(pady=20)

        # ==== ZONE TARGET ====
        self.target_frame = ctk.CTkFrame(self, width=512, height=512, corner_radius=5)
        self.target_frame.place(x=280, y=40)

        self.target_label = ctk.CTkLabel(self.target_frame, text="target")
        self.target_label.place(relx=0.5, rely=0.5, anchor="center")

        # ==== ZONE SOURCE ====
        self.source_frame = ctk.CTkFrame(self, width=512, height=512, corner_radius=5)
        self.source_frame.place(x=900, y=40)

        self.source_label = ctk.CTkLabel(self.source_frame, text="source")
        self.source_label.place(relx=0.5, rely=0.5, anchor="center")

        # ==== ZONE LOSS (MATPLOTLIB) ====
        loss_frame = ctk.CTkFrame(self, width=640, height=250, corner_radius=5)
        loss_frame.place(x=280, y=600)

        self.fig, self.ax = plt.subplots(figsize=(11.4, 2.5), constrained_layout=True)
        self.ax.set_facecolor("#ffffff")  # gris clair
        self.fig.patch.set_facecolor("#c4c4c4")  # fond de la figure

        # Grille
        self.ax.grid(True, color="#c4c4c4", linestyle='-', linewidth=1)  # grille blanche

        # Titres et labels
        self.ax.set_title("Loss", fontsize=12, fontweight="bold", color="#333333")
        self.ax.set_xlabel("Iteration", fontsize=10, color="#333333")
        self.ax.set_ylabel("Value", fontsize=10, color="#333333")


        self.ax.plot(self.loss, color="black", linewidth=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=loss_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==== ACTIONS ====

    def update_alpha1(self, value):
        self.alpha[16] = float(value)
        self.label_alpha1.configure(text=f"α1 = {value:.1f}")
        self.model.set_alphas(self.alpha)

    def update_alpha2(self, value):
        self.alpha[64] = float(value)
        self.label_alpha2.configure(text=f"α2 = {value:.1f}")
        self.model.set_alphas(self.alpha)

    def update_alpha3(self, value):
        self.alpha[512] = float(value)
        self.label_alpha3.configure(text=f"α3 = {value:.1f}")
        self.model.set_alphas(self.alpha)


    def main_loop(self):
        if self.iter == 0:
            self.model.random_start(50)
        
        loss, img = self.model.step()
            
        
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
            
        self.loss.append(loss)
        self.ax.clear()
        self.ax.plot(self.loss, color="black", linewidth=2)
        self.ax.set_title("loss")
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("value")
        self.ax.grid(True, color="#c4c4c4", linestyle='-', linewidth=1)  # grille blanche
        self.canvas.draw()
        img_tk = ctk.CTkImage(img, size=(512,512))
        self.source_label.configure(image=img_tk, text="")
        self.source_label.image = img_tk
            
        self.iter += 1
        if self.running:
            self.z = img
            self.after(1, self.main_loop)
        else:
            self.slider1.configure(state="normal")
            self.slider2.configure(state="normal")
            self.slider3.configure(state="normal")

    def load_image(self):
        if not self.running:
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
            )
            if not path:
                return

            # Charger et redimensionner
            self.image = Image.open(path).resize((512, 512))
            img_tensor = utils.load_img_tensor(path, self.device).unsqueeze(0)
            self.model = train.Inversor(img_tensor)

            # Créer un CTkImage
            self.ctk_image = ctk.CTkImage(self.image, size=(512, 512))

            # Afficher dans le label
            self.target_label.configure(image=self.ctk_image, text="")

            self.iter = 0
            self.loss = []
            

    def inv_action(self):
        if not self.running and self.iter == 0:
            self.slider1.configure(state="disabled")
            self.slider2.configure(state="disabled")
            self.slider3.configure(state="disabled")
            self.running = True
            self.after(1, self.main_loop)

    def qual_action(self):
        if not self.running and self.loss:
            print(f"La qualité de la reproduction est de : {self.loss[-1]}")

    def resume(self):
        if self.iter > 0 and not self.running:
            self.running = True
            self.after(1, self.main_loop)

    def stop_action(self):
        if self.iter > 0:
            self.running = False

if __name__ == "__main__":
    app = App()
    app.mainloop()

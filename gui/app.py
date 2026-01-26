import tkinter as tk
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
scriptpath = "../"
sys.path.append(os.path.abspath(scriptpath))
import core.train as train

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.running = False
        self.loss = []

        self.title("GUI Inversion")
        self.geometry("1500x900")
        self.configure(bg="#00b050")  # fond vert

        self.image = None
        self.photo = None
        
        self.model = train.init_model()

        self.build_ui()
        
        self.z0,_ = self.model.buildNoiseData(1)
        
        self.iter = 0

    def build_ui(self):        
        # ==== COLONNE BOUTONS ====
        btn_frame = tk.Frame(self, bg="#00b050")
        btn_frame.place(x=20, y=40, width=120, height=300)

        tk.Button(btn_frame, text="Load", command=self.load_image, width=10).pack(pady=10)
        tk.Button(btn_frame, text="Inv", command=self.inv_action, width=10).pack(pady=20)
        tk.Button(btn_frame, text="Qual", command=self.qual_action, width=10).pack(pady=20)

        # ==== ZONE TARGET ====
        self.target_frame = tk.Frame(self, bg="#2d4f1a", highlightbackground="black", highlightthickness=2)
        self.target_frame.place(x=180, y=40, width=512, height=512)

        self.target_label = tk.Label(self.target_frame, text="target", fg="white", bg="#2d4f1a")
        self.target_label.place(relx=0.5, rely=0.5, anchor="center")

        # ==== ZONE SOURCE ====
        self.source_frame = tk.Frame(self, bg="#2d4f1a", highlightbackground="black", highlightthickness=2)
        self.source_frame.place(x=800, y=40, width=512, height=512)

        self.source_label = tk.Label(self.source_frame, text="source", fg="white", bg="#2d4f1a")
        self.source_label.place(relx=0.5, rely=0.5, anchor="center")

        # ==== ZONE LOSS (MATPLOTLIB) ====
        loss_frame = tk.Frame(self, bg="#2d4f1a", highlightbackground="black", highlightthickness=2)
        loss_frame.place(x=400, y=600, width=640, height=250)

        self.fig, self.ax = plt.subplots(constrained_layout=True)
        self.ax.set_title("loss")
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("value")

        self.ax.plot(self.loss, color="black", linewidth=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=loss_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==== ACTIONS ====
    def load_image(self):
        if not self.running:
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
            )
            if not path:
                return
    
            self.image = Image.open(path).resize((512, 512))
            self.photo = ImageTk.PhotoImage(self.image)
    
            self.target_label.config(image=self.photo, text="")
    
            # garder une référence
            self.target_label.image = self.photo

    def inv_action(self):
        if self.image:
            if self.iter == 0:
                img, loss, self.running, self.z_new = train.inv(self.image, self.model, self.z0)
            else:
                img, loss, self.running, self.z_new = train.inv(self.image, self.model, self.z_new)
            img = ((img+1)/2).clamp(0,1)
            img = img.permute(1,2,0).cpu().numpy()
            img = (img*255).astype(np.uint8)
            img = Image.fromarray(img)
            self.loss.append(loss)
            self.ax.plot(self.loss, color="black", linewidth=1)
            self.canvas.draw()
            img_tk = ImageTk.PhotoImage(img)
            self.source_label.config(image=img_tk, text="")
            self.source_label.image = img_tk
            self.iter += 1
            if self.running:
                self.z = img
                self.after(1, self.inv_action)

    def qual_action(self):
        if not self.running:
            print(f"La qualité de la reproduction est de : {self.loss[-1]}")


if __name__ == "__main__":
    app = App()
    app.mainloop()


from tkinter import *
from tkinter import ttk, colorchooser
from PIL import Image
import io
import numpy as np
import torch
from diffusion_img2img import StableDiffusionImg2ImgPipeline

import matplotlib.pyplot as plt


class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=True).to("cuda")

        self.prompt_input = Text(parent, height = 3, width=50)
        self.prompt_input.grid(column=0, row=3, sticky=E)


        self.prompt_label = Label(parent, text="Image Prompt")
        self.prompt_label.grid(column=0, row=3, sticky=W)
        self.brush_color = "black"

        self.bind("<Button-1>", self.save_posn)
        self.bind("<B1-Motion>", self.add_line)
        self.pen_size = DoubleVar()

    def change_brush_color(self):
        self.brush_color = colorchooser.askcolor(title="Select brush color")[-1]

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        #self.create_line((self.lastx, self.lasty, event.x, event.y), width=10)
        self.create_oval((self.lastx, self.lasty, event.x, event.y), fill=self.brush_color, outline=self.brush_color, width=self.pen_size.get())
        self.save_posn(event)
    
    def save(self):
        ps = self.postscript(colormode='color')
        drawn_img = Image.open(io.BytesIO(ps.encode('utf-8')))
        #generator = torch.Generator("cuda").manual_seed(1024)
        out = self.model([self.prompt_input.get(1.0, "end-1c")], init_image=drawn_img)["sample"][0]
        fig, axs = plt.subplots(1)
        fig.patch.set_alpha(0.0)
        axs.imshow(out)
        plt.box(False)
        axs.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

    def clear(self):
        self.delete("all")

root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

sketch = Sketchpad(root, width=512, height=512, highlightthickness=1, highlightbackground="black")
sketch.grid(column=0, row=0, sticky=(N, W, E, S))

button = Button(root, text="Show Drawing",command=sketch.save)

button.grid(column=0, row=1, sticky= E)

clear_button = Button(root, text="Clear",command=sketch.clear)
clear_button.grid(column=1, row=1, sticky= E)

brush_size_label = Label(root, text="Brush Size")
brush_size_label.grid(column=0, row=2, sticky=W)

pen_slider = Scale(root, from_=1, to=50, orient="horizontal", variable=sketch.pen_size)
pen_slider.grid(column=0, row=2, sticky=S)

color_button = Button(root, text="Brush Color", command=sketch.change_brush_color)
color_button.grid(column=1, row=0, sticky= N)




root.mainloop()

import gradio as gr
import torch
import numpy as np
import cv2

from core.train import Inversor  # adapte si le fichier a un autre nom

device = "cpu"


# -------- Utils --------
def preprocess_image(img):
    """
    img: numpy uint8 [H,W,3] from gradio
    return: torch float [1,3,H,W] in [0,1]
    """
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


# -------- Gradio callbacks --------
def init_inversor(img):
    img = preprocess_image(img)
    inversor = Inversor(img)
    return inversor, None, "Inversor initialisé"


def random_start(inversor, n_random):
    inversor.random_start(n_random)
    return inversor, "Random start effectué"


def run_optimization(inversor, n_steps):
    """
    Streaming optimization
    """
    for i in range(n_steps):
        loss, gen_img = inversor.step()
        yield (
            gen_img,
            f"Step {i+1}/{n_steps} | Loss: {loss:.4f}"
        )


def set_alphas(inversor, alpha_64, alpha_512):
    inversor.set_alphas({64: alpha_64, 512: alpha_512})
    return inversor


# -------- UI --------
with gr.Blocks() as demo:
    gr.Markdown("## Face Inversion")

    inversor_state = gr.State()

    with gr.Row():
        input_image = gr.Image(label="Image cible", type="numpy")
        output_image = gr.Image(label="Image générée")

    status = gr.Textbox(label="Status")

    with gr.Row():
        init_btn = gr.Button("Initialiser")
        random_btn = gr.Button("Random Start")

    n_random = gr.Slider(1, 200, value=50, step=1, label="Random starts")

    n_steps = gr.Slider(1, 5000, value=1000, step=10, label="Steps d'optimisation")

    with gr.Row():
        alpha_64 = gr.Slider(0.0, 1.0, value=0.2, label="Alpha 64")
        alpha_512 = gr.Slider(0.0, 1.0, value=0.8, label="Alpha 512")

    run_btn = gr.Button("Lancer l'optimisation")

    # -------- Bindings --------
    init_btn.click(
        fn=init_inversor,
        inputs=input_image,
        outputs=[inversor_state, output_image, status]
    )

    random_btn.click(
        fn=random_start,
        inputs=[inversor_state, n_random],
        outputs=[inversor_state, status]
    )

    alpha_64.change(
        fn=set_alphas,
        inputs=[inversor_state, alpha_64, alpha_512],
        outputs=inversor_state
    )

    alpha_512.change(
        fn=set_alphas,
        inputs=[inversor_state, alpha_64, alpha_512],
        outputs=inversor_state
    )

    run_btn.click(
        fn=run_optimization,
        inputs=[inversor_state, n_steps],
        outputs=[output_image, status]
    )

demo.launch()

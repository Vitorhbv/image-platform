from huggingface_hub import InferenceClient
import os
from PIL import Image
import gradio as gr

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN"),
)
def generate_image_flux(text, negative_prompt, size,seed) -> gr.update:
    width = int(size.split("x")[0])
    height = int(size.split("x")[1])
    # output is a PIL.Image object
    image = client.text_to_image(
        text,
        model="black-forest-labs/FLUX.1-dev",
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed

    )
    return gr.update(value=image, width=width, height=height)

# image = generate_image_flux("A beautiful landscape", "ugly", "512x512", 12)
# image.save("output.png")
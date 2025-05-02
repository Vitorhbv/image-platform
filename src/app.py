import gradio as gr
import openai
import os
from PIL import Image
import base64
import io

api_key = os.environ.get('OPENAI_API_KEY')
api_base = "http://localhost:4000/v1"

client = openai.OpenAI(api_key=api_key, base_url=api_base)

#get the list of models
# models = client.models.list()
# print(models)

def ask_llava(image: Image.Image, question: str):
    """
    Send <image> + <question> to LLaVA and return the model's answer string.
    """
    # encode image → base64 → data-URI (OpenAI vision format)
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model="llava",              
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",       "text": question},
                    {"type": "image_url",  "image_url": {"url": data_uri}},
                ],
            }
        ],
    )

    return resp.choices[0].message.content.strip()

with gr.Blocks() as demo:
    gr.Markdown("### Ask LLaVA about an image")
    img   = gr.Image(type="pil", label="Upload or paste an image")
    ques  = gr.Textbox(label="Question", value="What’s in this picture?")
    btn   = gr.Button("Ask")
    out   = gr.Markdown()

    btn.click(ask_llava, inputs=[img, ques], outputs=out)

demo.launch()

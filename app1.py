import gradio as gr
from PIL import Image
from models.model import generate_image_flux

flux = "black-forest-labs/FLUX.1-dev"
MAX_ITEMS = 8


def submit_flux(prompt, negative_prompt, size, seed, history):
    pil_image = generate_image_flux(prompt, negative_prompt, size, seed)["value"]   # gera imagem
    pil_image = pil_image.convert("RGB")                                             # PIL.Image
    # empilha a nova imagem na frente (mantendo limite)
    history.insert(0, pil_image)
    if len(history) > MAX_ITEMS:
        history.pop()

    return gr.update(value=history, visible=True)

    
with gr.Blocks(fill_width=True, css="""
    #gal_hist {height: 512px;}
""") as app:
    gr.Markdown( """
                <div style="text-align: left; margin: 10 auto; max-width: 800px;">
                    <h1 style="font-size: 20px; color: #FF6E00;">Plataforma de Geração de Imagens</h1>
                </div>
                """)
    
     # Estado compartilhado → lista com todas as imagens geradas
    hist_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=10):
            gallery = gr.Gallery(
                label="Histórico de Imagens",
                elem_id="gal_hist",
                show_label=False,
                allow_preview=True,
                visible=True,
                container=True,
                preview=True,
                columns=4,
                rows=2,
                object_fit="contain",
                format="png"
            )
            with gr.Row():
                text= gr.Textbox(show_label=False,
                           placeholder="Digite seu prompt de texto aqui",
                           scale=8,
                           interactive=True,
                           container=False)
                btn = gr.Button("Gerar",size="lg",min_width=20,scale=1)
        with gr.Column(scale=3):
            model = gr.Dropdown(choices=[flux],label="Modelos")
            size = gr.Dropdown(choices=[
                    "320x320",
                    "480x320",
                    "640x320",
                    "800x320",
                    "960x480"
            ],
            label="Tamanho da Imagem")   
            negative_prompt = gr.Textbox(show_label=False, container=False, placeholder="Prompt negativo",scale=4)
               
            seed = gr.Slider(label="Ruído",
                             minimum=1,
                             maximum=100000,
                             step=1,
                             value=12,
                             interactive=True,
                             info="Ruído inicial do modelo. Para cada valor de ruído diferente, o modelo produzirá uma imagem diferente.",
                             scale=1)
                
    text.submit(fn=submit_flux, inputs=[text,negative_prompt,size,seed, hist_state], outputs=gallery, api_name="generate_image")
    btn.click(fn=submit_flux, inputs=[text,negative_prompt,size,seed, hist_state], outputs=gallery, api_name=False)

    #gallery.select(fn=show_history, inputs=[hist_state], outputs=output_image)

app.launch(server_name="0.0.0.0", server_port=7860)
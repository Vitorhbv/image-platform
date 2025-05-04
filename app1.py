import gradio as gr
from PIL import Image
from models.model import generate_image_flux

flux = "black-forest-labs/FLUX.1-dev"
MAX_COLS = 4
# ────────────────────────────────────────────────────────────────
# utilitário de fila
# ────────────────────────────────────────────────────────────────
def push_history(hist: list[Image.Image], new_img: Image.Image, limit: int = MAX_COLS):
    """
    • Se ainda não atingiu o limite → põe a nova imagem na frente.
    • Caso contrário:
        – descarta o item mais antigo (último da lista)
        – insere a nova na frente.
      (o “segundo mais antigo” já fica naturalmente na ponta direita)
    """
    if len(hist) >= limit:
        hist.pop()              # descarta o mais velho (último)
    hist.insert(0, new_img)      # nova imagem vira a primeira

def submit_flux(text, negative_prompt, size, seed, history):
    result = generate_image_flux(text, negative_prompt, size, seed)
    main_img = result["value"]

    push_history(history, main_img)            # aplica a fila
    gal_update = gr.update(value=history, visible=True)      # mostra/actualiza

    return result, gal_update

def show_history(evt: gr.SelectData, history):
    idx = evt.index
    return gr.update(value=history[idx])
    
with gr.Blocks() as app:
    gr.Markdown( """
                <div style="text-align: left; margin: 10 auto; max-width: 800px;">
                    <h1 style="font-size: 20px; color: #FF6E00;">Plataforma de Geração de Imagens</h1>
                </div>
                """)
    
     # Estado compartilhado → lista com todas as imagens geradas
    hist_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=10):
            output_image = gr.Image(
                label="Output Image",
                height=512
            )
             # HISTÓRICO (fica logo abaixo da imagem)
            gallery = gr.Gallery(
                label="Histórico de Imagens",
                show_label=False,
                columns=MAX_COLS,
                height=200,
                allow_preview=False,
                visible=False,
                container=False,
                
            )
            with gr.Row():
                text= gr.Textbox(show_label=False,
                           placeholder="Digite seu prompt de texto aqui",
                           scale=8,
                           interactive=True,
                           container=False)
                btn = gr.Button("Gerar",size="lg",min_width=20,scale=1)
        with gr.Column(scale=3):
            with gr.Row():
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
                
    text.submit(fn=submit_flux, inputs=[text,negative_prompt,size,seed, hist_state], outputs=[output_image, gallery], api_name="generate_image")
    btn.click(fn=submit_flux, inputs=[text,negative_prompt,size,seed, hist_state], outputs=[output_image, gallery], api_name=False)

    gallery.select(fn=show_history, inputs=[hist_state], outputs=output_image)

app.launch(server_name="0.0.0.0", server_port=7860)
#libraries
from dotenv import load_dotenv
# import os
from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
import gradio as gr
import time

# Load api key from env
load_dotenv("apis.env")
# hf_api_key = os.environ['HF_API_KEY']

# CSS Template 
theme = gr.themes.Base(
    primary_hue="rose",
).set(
    body_background_fill='*neutral_50',
    body_text_color='*neutral_500',
    body_text_weight='300',
    background_fill_primary='*neutral_50',
    background_fill_secondary='*primary_50',
    border_color_primary='*primary_400',
    color_accent_soft='*primary_300',
    link_text_color='*primary_300',
    link_text_color_active='*neutral_300',
    link_text_color_hover='*primary_100',
    link_text_color_visited='*neutral_400',
    code_background_fill='*primary_200',
    button_secondary_background_fill='*neutral_100',
    button_secondary_border_color='*neutral_900',
    button_secondary_text_color='*primary_400',
    button_cancel_background_fill='*primary_600',
    button_cancel_background_fill_hover='*primary_700',
    button_cancel_text_color='*neutral_50',
    slider_color='*primary_500'
)

def test(message):
    for i in range(len(message)):
        time.sleep(0.04)
        yield message[: i + 1]

def respond(msg, pdf, temperature, max_tokens):
    documents = SimpleDirectoryReader(
        input_files=[pdf]
    ).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=10)

    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True)
    
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(msg)
    return response

def process_file(uploaded_file):
    if uploaded_file is not None:
        filename = uploaded_file.name
        return filename
    return "No file uploaded."
    
def main():
    with gr.Blocks(theme=theme) as demo: 
        gr.Markdown("# TFM: RAG App\U0001F50E") #\U0001F468\u200D\U0001F393
        gr.Markdown("This RAG app is designed to generate augmented responses from a PDF file which contains text.")

        file_input = gr.File(
            # value="C:/Users/Nacho/Documents/MASTER/TFM/tfm-thesis/attention-is-all-you-need.pdf",
            file_types=[".pdf"],
            every=process_file)
        
        output_label = gr.Textbox(visible=False)

        file_input.change(process_file, inputs=file_input, outputs=output_label)

        msg = gr.Textbox(label="Prompt", value="What is the main topic in the text?")
        with gr.Accordion(label="Advanced options",open=False):
            # temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1.0, value=0.2, step=0.1, info="Regulates the creativity of the answers", visible=False)
            # max_tokens = gr.Slider(label="Max tokens", value=64, maximum=256, minimum=8, step=1, info="Regulates the length of the answers", visible=False)
            
            creativity = gr.Radio(
                label="Creativity",
                interactive=True,
                choices=[("None", 0),("Medium", 0.5),("High", 1)],
                value=("None", 0),
                info="Regulates the creativity of the answers"
                )
            length = gr.Radio(
                label="Output length",
                interactive=True,
                choices=[("Short", 64), ("Medium", 512), ("Large", 1024)],
                value=("Short", 64),
                info="Regulates the length of the answers"
                )
        
        completion = gr.Textbox(label="Response")
        # stream_out = gr.Textbox(visible=False)
        btn = gr.Button("Submit", variant="primary")
        clear = gr.ClearButton(components=[msg, completion], value="Clear console", variant="stop")
        
        btn.click(respond, inputs=[msg, output_label, creativity, length], outputs=[completion]) #, temperature, max_tokens
        msg.submit(respond, inputs=[msg, output_label, creativity, length], outputs=[completion]) #, temperature, max_tokens

        # completion.change(fn=test, inputs=stream_out)

        gr.Markdown("\U0001F6E0Created by Ignacio Ojeda Sánchez ", header_links=True)
    gr.close_all()
    demo.queue().launch(share=True)
    print("\n\\U0001F468\u200D\U0001F393")
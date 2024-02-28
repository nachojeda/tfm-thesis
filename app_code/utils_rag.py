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
        model="gpt-3.5-turbo", #-instruct
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

def update_file_input(choice):
    # This function updates the file input component based on the choice
    # For simplicity, this example doesn't directly upload the chosen file
    # but updates the label to indicate which file is to be uploaded.
    # You might need to adjust it to fit your actual file handling logic.
    if choice == "file1.pdf":
        return "Upload file1.pdf", True
    elif choice == "file2.pdf":
        return "Upload file2.pdf", True
    else:
        return "Upload a PDF file", False  # Default state
    
def main():
    with gr.Blocks(theme=theme) as app: 
        gr.Markdown("# TFM: RAG App\U0001F50E") #\U0001F468\u200D\U0001F393
        gr.Markdown("This RAG app is designed to generate augmented responses from a PDF file which contains text.")
        # with gr.Row():
        #     pdf_dropdown = gr.Dropdown(label="Select a PDF file", choices=["Select a PDF", "pdf1.pdf", "pdf2.pdf", "pdf3.pdf"], value="Select a PDF")
        #     file_upload = gr.File(label="Or upload a PDF file")
            
        # pdf_label = gr.Label()  # To display selected PDF.
        # upload_label = gr.Label()  # To display uploaded file info.
        
        # pdf_dropdown.change(handle_pdf_or_upload, inputs=[pdf_dropdown, file_upload, pdf_label, upload_label], outputs=[pdf_label, upload_label])
        # file_upload.change(handle_pdf_or_upload, inputs=[pdf_dropdown, file_upload, pdf_label, upload_label], outputs=[pdf_label, upload_label])
        

        file_input = gr.File(
            value = "C:/Users/Nacho/Documents/MASTER/TFM/tfm-thesis/attention-is-all-you-need.pdf",
            scale = 1,
            label = "Upload a PDF file",
            file_types=[".pdf"],
            every=process_file
        )

        # files = [
        #    "attention-is-all-you-need.pdf",
        #     "Music Genre Classification A Comparative Study Between Deep Learning and Traditional Machine Learning Approaches.pdf"
        # ]

        # with gr.Row():
        #     # Dropdown for selecting the file to upload
        #     file_selector = gr.Dropdown(
        #         choices=files,
        #         label="Select a PDF file to upload"
        #     )
        
        #     # File input component, initially not showing any specific file choice
        #     file_input = gr.File(
        #         label="Upload a PDF file",
        #         file_types=[".pdf"],
        #         visible=False  # Initially hidden, shown after selection
        #     )

        # file_selector.change(update_file_input, inputs=[file_selector], outputs=[file_input.label, file_input.visible])

        # output_label = gr.Textbox(visible=False)

        # file_input.change(process_file, inputs=file_input, outputs=output_label)

        msg = gr.Textbox(label="Prompt", placeholder="Insert a query, for example: Elaborate on the transformer")
        with gr.Accordion(label="Advanced options",open=False):
            # temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1.0, value=0.2, step=0.1, info="Regulates the creativity of the answers", visible=False)
            # max_tokens = gr.Slider(label="Max tokens", value=64, maximum=256, minimum=8, step=1, info="Regulates the length of the answers", visible=False)
            
            creativity = gr.Radio(
                label="Creativity",
                interactive=True,
                choices=[("None", 0),("Medium", 0.5),("High", 1)],
                value=0,
                info="Regulates the creativity of the answers"
                )
            length = gr.Radio(
                label="Output length",
                interactive=True,
                choices=[("Short", 32), ("Medium", 1024), ("Large", 2048)],
                value=32,
                info="Regulates the maximum length of the answers"
                )
        
        completion = gr.Textbox(label="Response")
        # stream_out = gr.Textbox(visible=False)
        btn = gr.Button("Submit", variant="primary")
        clear = gr.ClearButton(components=[msg, completion], value="Clear console", variant="stop")
        
        btn.click(respond, inputs=[msg, file_input, creativity, length], outputs=[completion]) #, temperature, max_tokens
        msg.submit(respond, inputs=[msg, file_input, creativity, length], outputs=[completion]) #, temperature, max_tokens

        # completion.change(fn=test, inputs=stream_out)

        gr.Markdown("\U0001F6E0Developt by Ignacio Ojeda Sánchez ", header_links=True)
    gr.close_all()
    app.queue().launch(share=True)
    print("\n\nFinished")
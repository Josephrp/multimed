import gradio as gr


def process_speech(audio):
    return f"""audio {audio}
type {type(audio)}"""


with gr.Blocks(theme='ParityError/Anime') as iface : 
    audio_input = gr.Audio(label="talk in french")
    audio_output = gr.Markdown(label="output text")
    audio_button = gr.Button("process audio")
    audio_button.click(process_speech, inputs=audio_input, outputs=audio_output)


iface.launch(show_error=True)
import gradio as gr
from hand_tracking_modul import main

def video_identity(video):
    return main(video)


demo = gr.Interface(video_identity,
                    gr.Video(),
                    "playable_video",
                    )

if __name__ == "__main__":
    demo.launch(share=True)

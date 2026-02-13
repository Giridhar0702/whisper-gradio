import whisper
import gradio as gr

model = whisper.load_model("small")

def transcribe(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return result.text

gr.Interface(
    title="Speech-to-Text App",
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),  # ✅ FIXED
    outputs=gr.Textbox(label="Transcribed Text"),
).launch()

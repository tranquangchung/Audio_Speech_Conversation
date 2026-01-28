# Spoken Dialogue: Speech-to-Speech Conversation

This repository contains the implementation of a **Speech-to-Speech (S2S) Dialogue System**.

---

## 📥 Model Checkpoints

The pre-trained and fine-tuned checkpoints for this dialogue system are available on Hugging Face:

**Download Link:** [tranquangchung/qwen2-audio-dialogue](https://huggingface.co/tranquangchung/qwen2-audio-dialogue)

You can clone the model using:
```bash
git lfs install
git clone [https://huggingface.co/tranquangchung/qwen2-audio-dialogue](https://huggingface.co/tranquangchung/qwen2-audio-dialogue)
```

## 🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.10+ and the necessary audio processing libraries installed:

```Bash
pip install torch torchaudio transformers accelerate librosa
```
2. To test the dialogue system with real-world audio samples ("in-the-wild"), run the provided inference script:

```Bash
python test_dialogue_inthewild.py
```
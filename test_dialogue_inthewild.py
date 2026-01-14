import os
import glob
import torch
import librosa
from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from models.processing_qwen2_audio import Qwen2AudioProcessor
import logging
import pdb

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

checkpoint = "result_1/dialogueV3/qwen2llm_audio_dialogue_1e4_bs16_0.5B_correct/checkpoint_iter_25000"
print(f"Loading checkpoint: {checkpoint}")

processor = Qwen2AudioProcessor.from_pretrained(checkpoint)
model = Qwen2AudioForConditionalGeneration.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Ensure we use the correct End-Of-Sequence token
IM_END_ID = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

# --- 🆕 Data Configuration for "In the Wild" ---
audio_folder = "/home/ldap-users/s2220411/Code/new_explore_tts/CosyVoice/LLM_Question"
print(f"Scanning directory: {audio_folder}")

# Get all .wav files and sort them (01.wav, 02.wav, etc.)
audio_files = sorted(glob.glob(os.path.join(audio_folder, "*.wav")))

if not audio_files:
    print("❌ No .wav files found in the directory!")
    exit()

print(f"Found {len(audio_files)} files. Starting Inference...")

text_sources = [
    "If you had to live in a video game universe for a year, which one would you choose?",
    "Would you rather be able to speak every language fluently or play every musical instrument perfectly?",
    "If animals could talk, which species do you think would be the rudest?",
    "If you found a suitcase with $1 million inside but no ID, would you keep it or turn it in?",
    "If you were a ghost, who would you haunt and how?",
    "Would you rather travel 100 years into the past or 100 years into the future?",
    "If you could only eat one cuisine (Italian, Japanese, Mexican, etc.) for the rest of your life, what would it be?",
    "If you were forced to participate in the Olympics, which sport would you have the best chance of not embarrassing yourself in?",
    "If you could cure one disease or solve one world problem, which would you pick?",
    "If you had to delete all apps from your phone except for three, which ones would you keep?",
    "What is the capital city of Vietnam?",
    "Vietnam is consistently one of the world's top two exporters of which caffeinated commodity?",
    "What is the ancient name of Hanoi, which translates to Ascending Dragon",
    "In Vietnamese culture, what are the specific flowers used to decorate homes during the Tết holiday (Lunar New Year) in the North versus the South?"
]

# --- Inference Loop ---
for index, audio_path in enumerate(audio_files):
    filename = os.path.basename(audio_path)

    # Load Audio
    # Qwen2-Audio expects 16k usually, but using processor.feature_extractor.sampling_rate is safer
    target_sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=target_sr)

    # --- 1. Construct Prompt ---
    # Since we don't have a text "question" from JSON, we use a generic instruction.
    DEFAULT_INSTRUCTION = "You are a virtual assistant, answering questions and providing information in a clear and polite manner."

    text_prompt = (
        f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"<|im_start|>user\n"
        f"Audio content provided. {DEFAULT_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # --- 2. Process Inputs ---
    inputs = processor(
        text=text_prompt,
        audio=audio,
        return_tensors="pt",
        sampling_rate=target_sr,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- 3. Generate ---
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=IM_END_ID,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True
        )

    # --- 4. Decode ---
    input_len = inputs['input_ids'].size(1)
    generated_ids_trimmed = generate_ids[:, input_len:]
    pdb.set_trace()

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("*" * 30)
    print(f"File: {filename}")
    print("Question: ", text_sources[index])
    print(f"Generated Answer: {result.strip()}")
    print("-" * 10)
    print("*" * 30)
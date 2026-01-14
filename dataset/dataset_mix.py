import torch
import json
import librosa
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
import torch.nn.functional as F
from models.processing_qwen2_audio import Qwen2AudioProcessor
import termplotlib as tpl
import numpy as np
import utils
import pdb

class AudioSpeechDataset(Dataset):
    def __init__(
        self,
        json_files: List[str],
        processor: Qwen2AudioProcessor,
        max_length: int = 16000 * 10,
        sampling_rate: int = 16000,
    ):
        super().__init__()
        self.json_files = [Path(p) for p in json_files]
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate

        self.data = self._load_json_files()
        if len(self.data) == 0:
            raise ValueError("No valid samples after loading JSON files")

    def _load_json_files(self) -> List[Dict]:
        all_data = []
        audio_lengths = []
        # required = {"audio", "text", "prompt", "task"}
        for fp in self.json_files:
            if not fp.exists():
                print(f"[WARN] JSON not found: {fp}")
                continue
            try:
                items = json.loads(fp.read_text(encoding="utf-8"))
                for item in items:
                    if not Path(item["audio"]).exists():
                        continue
                    all_data.append(item)
            except Exception as e:
                print(f"[WARN] Load {fp} failed: {e}")
        return all_data

    def plot_histogram(data, color=None):
        if color is None:
            color = utils.GREEN
        print(f"{color}Data with target unit length distribution")
        hist, bins = np.histogram(data, bins=20)
        # Create and show the plot
        fig = tpl.figure()
        fig.hist(hist, bins, force_ascii=True, orientation="horizontal")
        fig.show()
        print(f"{utils.RESET}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[dict]:
        try:
            entry = self.data[idx]
            audio_path = entry["audio"]
            text = entry["answer"]
            prompt = entry["prompt"]
            question = entry["question"]
            task = entry["task"]
            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
            if audio.shape[0] > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = librosa.util.pad_center(audio, size=self.max_length)

            # <|audio_bos|><|AUDIO|><|audio_eos|> + [Chỉ dẫn hệ thống]
            user_part = (
                f"<|audio_bos|><|AUDIO|><|audio_eos|>"
                f"<|im_start|>user\n"
                f"Audio content provided. {prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            pdb.set_trace()

            # 3. Tokenize (Truyền audio vào)
            full_text = f"{user_part}{text.strip()}<|im_end|>"
            inputs = self.processor(
                text=full_text,
                audio=audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=False
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Tạo labels
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # Tính độ dài prompt để mask
            user_inputs = self.processor(
                text=user_part,
                audio=audio,  # Nếu nhánh Text thì audio=None
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=False
            )
            prompt_len = user_inputs["input_ids"].size(1)

            # Masking (Gán -100 cho phần Prompt)
            if prompt_len < labels.size(0):
                labels[:prompt_len] = -100
            else:
                return None  # Skip mẫu lỗi
            inputs["labels"] = labels
            inputs["text"] = full_text
            return inputs

        except Exception as e:
            print(f"Error processing sample {idx} ({self.data[idx]['audio']}): {e}")
            return None

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("All samples in batch are None.")

        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        input_features = [b["input_features"] for b in batch]
        feature_attention_mask = [b["feature_attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        texts = [b["text"] for b in batch]

        # text part
        max_txt = max(x.size(0) for x in input_ids)
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids = torch.stack([F.pad(x, (0, max_txt - x.size(0)), value=pad_id) for x in input_ids])
        attention_mask = torch.stack([F.pad(x, (0, max_txt - x.size(0)), value=0) for x in attention_mask])
        labels = torch.stack([F.pad(x, (0, max_txt - x.size(0)), value=-100) for x in labels])

        # audio part
        max_mel = max(x.size(-1) for x in input_features)
        input_features = torch.stack([F.pad(x, (0, max_mel - x.size(-1))) for x in input_features])
        feature_attention_mask = torch.stack([F.pad(x, (0, max_mel - x.size(0)), value=0) for x in feature_attention_mask])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
            # "texts": texts,
        }
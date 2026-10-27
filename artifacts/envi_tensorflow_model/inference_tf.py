"""Sinh bản dịch từng token, dùng encoder/decoder tách ra từ cùng một model."""
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from data import BOS, EOS, PAD, normalize_text
from model_tf import build_inference_models


def decode_token_ids(tokenizer, token_ids):
    if EOS in token_ids:
        token_ids = token_ids[:token_ids.index(EOS)]
    token_ids = [token for token in token_ids if token not in (PAD, BOS, EOS)]
    return normalize_text(tokenizer.decode(token_ids))


class Translator:
    def __init__(self, model, english_tokenizer, vietnamese_tokenizer, config):
        self.encoder, self.decoder = build_inference_models(model)
        self.english_tokenizer = english_tokenizer
        self.vietnamese_tokenizer = vietnamese_tokenizer
        self.config = config
        # Mỗi lần gọi chỉ chạy 1 bước decoder; tf.function giảm overhead khi dịch.
        self.encoder_step = tf.function(self.encoder, reduce_retracing=True)
        self.decoder_step = tf.function(self.decoder, reduce_retracing=True)

    def translate_ids(self, source_sequences):
        """Greedy decoding cho một batch; dừng mỗi câu khi gặp EOS."""
        source_batch = tf.keras.utils.pad_sequences(source_sequences, padding="post", value=PAD)
        memory, hidden, cell, source_mask = self.encoder_step(
            tf.convert_to_tensor(source_batch, dtype=tf.int32), training=False
        )
        batch_size = len(source_sequences)
        current_token = np.full((batch_size, 1), BOS, dtype=np.int32)
        finished = np.zeros(batch_size, dtype=bool)
        generated = [[] for _ in source_sequences]
        for _ in range(self.config["max_tokens"]):
            logits, hidden, cell, _ = self.decoder_step(
                [tf.convert_to_tensor(current_token), memory, hidden, cell, source_mask],
                training=False,
            )
            scores = logits.numpy()[:, 0, :]
            scores[:, [PAD, BOS]] = -np.inf
            next_tokens = np.argmax(scores, axis=-1).astype(np.int32)
            for index, token in enumerate(next_tokens):
                if not finished[index]:
                    generated[index].append(int(token))
                    finished[index] = token == EOS
            if finished.all():
                break
            current_token = np.where(finished, EOS, next_tokens)[:, None].astype(np.int32)
        return generated

    def translate(self, sentence):
        sentence = normalize_text(sentence)
        if not sentence:
            raise ValueError("Em hãy nhập một câu tiếng Anh.")
        if len(sentence.split()) > self.config["max_words"]:
            raise ValueError("Câu dài hơn phạm vi huấn luyện; hãy chia thành câu ngắn hơn.")
        source_ids = [BOS] + self.english_tokenizer.encode(sentence) + [EOS]
        if len(source_ids) > self.config["max_tokens"]:
            raise ValueError("Câu vượt giới hạn token; chương trình không tự cắt bỏ nội dung.")
        result = self.translate_ids([source_ids])[0]
        return decode_token_ids(self.vietnamese_tokenizer, result)


def load_translator(folder):
    folder = Path(folder)
    # Import model_tf đăng ký custom layers trước khi load_model.
    model = tf.keras.models.load_model(folder / "model.keras", compile=False)
    english = spm.SentencePieceProcessor(model_file=str(folder / "tokenizer_en.model"))
    vietnamese = spm.SentencePieceProcessor(model_file=str(folder / "tokenizer_vi.model"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    return Translator(model, english, vietnamese, config)


if __name__ == "__main__":
    import sys
    translator = load_translator(Path(__file__).parent)
    print(translator.translate(" ".join(sys.argv[1:])))

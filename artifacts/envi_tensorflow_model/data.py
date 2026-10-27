"""Tải và chuẩn bị dữ liệu Anh–Việt. Chỉ dùng thư viện chuẩn + SentencePiece."""
import hashlib
import html
import json
import random
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

import sentencepiece as spm

PAD, UNK, BOS, EOS = 0, 1, 2, 3

DATA_FILES = {
    "train": ("train-en-vi.tgz", "train.en", "train.vi",
              "e50f5de7e6625a7dd1fe5793595e973e87fe955768006b00b3e1ef7f18c7793a"),
    "validation": ("dev-2012-en-vi.tgz", "tst2012.en", "tst2012.vi",
                   "04a0b7ce677cea8ad9dac65bca9492db8183dcfefa458f820597dc46d7486dad"),
    "test": ("test-2013-en-vi.tgz", "tst2013.en", "tst2013.vi",
             "0bbf7cc1ddc4bed68946b005edf9471478445e7994586a277ad4a4b0bb24caa4"),
}
DATA_URL = "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text):
    """Giữ dấu tiếng Việt, chữ hoa và số; sửa khoảng trắng quanh dấu câu."""
    text = unicodedata.normalize("NFC", html.unescape(text.strip()))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    return text


def download_parallel_data(cache_folder):
    """Xác minh checksum và chỉ đọc hai tệp văn bản đã biết trong mỗi archive."""
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    raw_pairs = {}
    for split, (archive, english_file, vietnamese_file, expected_hash) in DATA_FILES.items():
        archive_path = cache_folder / archive
        if not archive_path.exists():
            temporary_path = archive_path.with_suffix(".tmp")
            with urllib.request.urlopen(DATA_URL + archive, timeout=90) as response:
                temporary_path.write_bytes(response.read())
            temporary_path.replace(archive_path)
        actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Tệp dữ liệu không đúng checksum: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive_file:
            english = archive_file.extractfile(english_file).read().decode("utf-8").splitlines()
            vietnamese = archive_file.extractfile(vietnamese_file).read().decode("utf-8").splitlines()
        if len(english) != len(vietnamese):
            raise ValueError(f"Số dòng Anh và Việt không khớp trong tập {split}")
        raw_pairs[split] = [(normalize_text(en), normalize_text(vi))
                            for en, vi in zip(english, vietnamese)]
    return raw_pairs

def remove_duplicates(pairs, forbidden_sources):
    seen = set(forbidden_sources)
    result = []
    for english, vietnamese in pairs:
        key = english.casefold()
        if english and vietnamese and key not in seen:
            result.append((english, vietnamese))
            seen.add(key)
    return result


def prepare_splits(raw_pairs, config):
    """Giữ split gốc; loại trùng nguồn giữa train/validation/test trước khi học."""
    test_sources = {en.casefold() for en, _ in raw_pairs["test"]}
    validation_sources = {en.casefold() for en, _ in raw_pairs["validation"]}
    pairs = {
        "test": remove_duplicates(raw_pairs["test"], set()),
        "validation": remove_duplicates(raw_pairs["validation"], test_sources),
        "train": remove_duplicates(raw_pairs["train"], test_sources | validation_sources),
    }
    random_generator = random.Random(config["seed"])
    for split in pairs:
        pairs[split] = [(en, vi) for en, vi in pairs[split]
                        if 1 <= len(en.split()) <= config["max_words"]
                        and (split != "train" or len(vi.split()) <= config["max_words"])]
        random_generator.shuffle(pairs[split])
        limit = config.get("limit_" + split, 0)
        if limit:
            pairs[split] = pairs[split][:limit]
    return pairs


def train_tokenizers(train_pairs, config, output_folder):
    """Hai tokenizer BPE chỉ học từ tập train, không dùng validation/test."""
    output_folder = Path(output_folder)
    tokenizers = []
    for language, column in [("en", 0), ("vi", 1)]:
        training_file = output_folder / f"tokenizer_training.{language}"
        training_file.write_text("\n".join(pair[column] for pair in train_pairs), encoding="utf-8")
        prefix = output_folder / f"tokenizer_{language}"
        spm.SentencePieceTrainer.train(
            input=str(training_file), model_prefix=str(prefix),
            vocab_size=config["vocab_size"], model_type="bpe", character_coverage=1.0,
            pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
            normalization_rule_name="identity", hard_vocab_limit=False,
            shuffle_input_sentence=False, num_threads=1, minloglevel=2,
        )
        training_file.unlink()
        tokenizers.append(spm.SentencePieceProcessor(model_file=str(prefix) + ".model"))
    return tokenizers


def encode_splits(pairs, english_tokenizer, vietnamese_tokenizer, config):
    """Không cắt câu âm thầm. Câu tham chiếu test luôn giữ nguyên."""
    encoded, kept_text = {}, {}
    for split, split_pairs in pairs.items():
        encoded[split], kept_text[split] = [], []
        for english, vietnamese in split_pairs:
            source_ids = [BOS] + english_tokenizer.encode(english) + [EOS]
            target_ids = [BOS] + vietnamese_tokenizer.encode(vietnamese) + [EOS]
            if len(source_ids) > config["max_tokens"]:
                continue
            if split == "train" and len(target_ids) > config["max_tokens"]:
                continue
            encoded[split].append((source_ids, target_ids))
            kept_text[split].append((english, vietnamese))
    keys = {split: {en.casefold() for en, _ in values} for split, values in kept_text.items()}
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["test"]
    assert not keys["validation"] & keys["test"]
    if any(not split for split in encoded.values()):
        raise ValueError("Có tập dữ liệu rỗng sau khi lọc.")
    return encoded, kept_text

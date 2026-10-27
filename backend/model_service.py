"""
Dịch vụ nạp model và thực hiện suy luận cho EnVi Lab.
Giữ nguyên tệp model gốc, đóng vai trò adapter xử lý xác thực dữ liệu,
kiểm tra giới hạn token/từ, đo thời gian suy luận và đảm bảo an toàn luồng.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Đảm bảo KERAS_HOME trỏ tới thư mục cục bộ trong workspace
if "KERAS_HOME" not in os.environ:
    local_keras = Path(__file__).parent / ".keras"
    local_keras.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(local_keras.resolve())

logger = logging.getLogger("envi_backend.model_service")

REQUIRED_FILES = [
    "model.keras",
    "tokenizer_en.model",
    "tokenizer_vi.model",
    "config.json",
]


class ModelService:
    def __init__(self, model_dir_path: Optional[str] = None):
        self.model_dir = self._resolve_model_dir(model_dir_path)
        self.is_ready = False
        self.missing_files: List[str] = []
        self.translator: Any = None
        self.config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._infer_lock = threading.Lock()

    def _resolve_model_dir(self, path_str: Optional[str]) -> Path:
        """Tìm đường dẫn model thực tế từ biến môi trường hoặc các vị trí mặc định."""
        candidates = []
        if path_str:
            candidates.append(Path(path_str))
        env_dir = os.environ.get("MODEL_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        
        base_dir = Path(__file__).parent.resolve()
        candidates.extend([
            base_dir / "artifacts" / "envi_tensorflow_model",
            base_dir.parent / "artifacts" / "envi_tensorflow_model",
            base_dir / "envi_tensorflow_model",
            base_dir.parent / "envi_tensorflow_model",
        ])

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved

        # Nếu chưa tìm thấy thư mục tồn tại, trả về ứng viên đầu tiên để kiểm tra thiếu tệp
        return (base_dir.parent / "artifacts" / "envi_tensorflow_model").resolve()

    def load_model(self) -> None:
        """Nạp model và tokenizer một lần khi backend khởi động."""
        logger.info(f"Đang kiểm tra thư mục model tại: {self.model_dir}")
        self.missing_files = []

        if not self.model_dir.exists():
            self.missing_files = list(REQUIRED_FILES)
            self.is_ready = False
            logger.warning(f"Thư mục model không tồn tại: {self.model_dir}")
            return

        for fname in REQUIRED_FILES:
            if not (self.model_dir / fname).exists():
                self.missing_files.append(fname)

        if self.missing_files:
            self.is_ready = False
            logger.warning(f"Các tệp model còn thiếu: {self.missing_files}")
            return

        try:
            # Thêm thư mục model vào sys.path để import đúng các module gốc (data, model_tf, inference_tf)
            model_dir_str = str(self.model_dir)
            if model_dir_str not in sys.path:
                sys.path.insert(0, model_dir_str)

            from inference_tf import load_translator  # type: ignore

            logger.info("Bắt đầu nạp TensorFlow model và SentencePiece tokenizers...")
            self.translator = load_translator(self.model_dir)
            self.config = self.translator.config

            # Nạp thông tin bổ sung (run_summary, evaluation, data_manifest)
            self.metadata = self._load_extra_metadata()

            # Warm-up suy luận 1 câu ngắn để khởi tạo TensorFlow graph
            logger.info("Chạy warm-up inference...")
            _ = self.translator.translate("Hello.")
            self.is_ready = True
            logger.info("Model EnVi đã sẵn sàng phục vụ suy luận.")
        except Exception as e:
            self.is_ready = False
            logger.exception(f"Lỗi khi nạp model: {e}")

    def _load_extra_metadata(self) -> Dict[str, Any]:
        """Đọc các tệp báo cáo kỹ thuật để cung cấp cho API model-info."""
        meta: Dict[str, Any] = {}
        files = {
            "evaluation": self.model_dir / "evaluation.json",
            "run_summary": self.model_dir / "run_summary.json",
            "data_manifest": self.model_dir / "data_manifest.json",
            "technical_checks": self.model_dir / "technical_checks.json",
        }
        for key, path in files.items():
            if path.exists():
                try:
                    meta[key] = json.loads(path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Không thể đọc {path.name}: {e}")
        return meta

    async def translate_text(self, text: str) -> Dict[str, Any]:
        """
        Xác thực dữ liệu đầu vào và thực hiện suy luận bất đồng bộ.
        Trả về dict chứa bản dịch, thời gian xử lý và số từ/token.
        """
        if not self.is_ready or self.translator is None:
            missing_msg = f" (thiếu: {', '.join(self.missing_files)})" if self.missing_files else ""
            raise RuntimeError(f"Hệ thống dịch chưa sẵn sàng{missing_msg}.")

        if not isinstance(text, str):
            raise ValueError("Đầu vào phải là chuỗi văn bản.")

        raw_trimmed = text.strip()
        if not raw_trimmed:
            raise ValueError("Vui lòng nhập một câu tiếng Anh cần dịch.")

        # Thư viện chuẩn hóa từ data.py
        from data import normalize_text, BOS, EOS  # type: ignore
        norm_sentence = normalize_text(raw_trimmed)
        if not norm_sentence:
            raise ValueError("Vui lòng nhập một câu tiếng Anh hợp lệ.")

        words = norm_sentence.split()
        max_words = self.config.get("max_words", 40)
        if len(words) > max_words:
            raise ValueError(
                f"Câu chứa {len(words)} từ, vượt quá giới hạn cho phép ({max_words} từ). "
                "Vui lòng rút gọn hoặc chia thành câu ngắn hơn."
            )

        # Kiểm tra token limit
        source_subwords = self.translator.english_tokenizer.encode(norm_sentence)
        source_ids = [BOS] + source_subwords + [EOS]
        max_tokens = self.config.get("max_tokens", 96)
        if len(source_ids) > max_tokens:
            raise ValueError(
                f"Câu được mã hóa thành {len(source_ids)} subword token, vượt quá giới hạn ({max_tokens} token). "
                "Chương trình không tự cắt bỏ nội dung; vui lòng rút gọn câu."
            )

        # Thực thi dịch an toàn với khóa và worker thread
        def _sync_translate(s: str) -> str:
            with self._infer_lock:
                return self.translator.translate(s)

        start_time = time.perf_counter()
        vietnamese_result = await asyncio.to_thread(_sync_translate, norm_sentence)
        elapsed_sec = time.perf_counter() - start_time

        return {
            "source_text": norm_sentence,
            "translated_text": vietnamese_result,
            "source_word_count": len(words),
            "source_token_count": len(source_ids),
            "latency_ms": round(elapsed_sec * 1000, 2),
            "latency_sec": round(elapsed_sec, 3),
        }

    def get_info(self) -> Dict[str, Any]:
        """Cung cấp thông tin kiến trúc, cấu hình và kết quả thực nghiệm từ các tệp báo cáo."""
        eval_data = self.metadata.get("evaluation", {})
        run_data = self.metadata.get("run_summary", {})
        manifest_data = self.metadata.get("data_manifest", {})

        return {
            "is_ready": self.is_ready,
            "missing_files": self.missing_files,
            "model_path": str(self.model_dir),
            "architecture": {
                "name": "EnVi BiLSTM with Bahdanau Attention",
                "framework": "TensorFlow / Keras 3 (Functional API)",
                "encoder": "Bidirectional LSTM (256 units mỗi chiều, tổng 512 units)",
                "decoder": "Unidirectional LSTM (512 units)",
                "attention": "Additive Bahdanau Attention (128 units)",
                "embedding_dim": self.config.get("embedding_dim", 256),
                "vocab_size": self.config.get("vocab_size", 8000),
                "tokenizer_type": "SentencePiece BPE (8.000 token/ngôn ngữ)",
                "total_parameters": eval_data.get("parameters", 9171136),
                "decoding_strategy": eval_data.get("decoding", "greedy"),
            },
            "limits": {
                "max_words": self.config.get("max_words", 40),
                "max_tokens": self.config.get("max_tokens", 96),
                "supported_language_pair": "English -> Vietnamese",
            },
            "dataset": {
                "name": manifest_data.get("dataset", "IWSLT'15 English-Vietnamese TED"),
                "train_pairs": manifest_data.get("used_counts", {}).get("train", 115154),
                "validation_pairs": manifest_data.get("used_counts", {}).get("validation", 1492),
                "test_pairs": manifest_data.get("used_counts", {}).get("test", 1177),
                "filtering": "Loại trùng lặp, lọc câu <= 40 từ và <= 96 subwords; tokenizer chỉ học từ train split",
            },
            "training": {
                "hardware": "Tesla T4 GPU (Google Colab)",
                "epochs_completed": run_data.get("epochs_completed", 19),
                "best_epoch": run_data.get("best_epoch", 14),
                "best_validation_token_nll": run_data.get("best_validation_token_nll", 2.6487),
                "batch_size": self.config.get("batch_size", 64),
                "learning_rate": self.config.get("learning_rate", 0.001),
                "dropout": self.config.get("dropout", 0.2),
            },
            "evaluation": {
                "test_sentences": eval_data.get("test_sentences", 1177),
                "bleu": round(eval_data.get("bleu", 24.14), 2),
                "chrf_pp": round(eval_data.get("chrf_pp", 43.05), 2),
                "bleu_signature": eval_data.get("bleu_signature", ""),
                "chrf_signature": eval_data.get("chrf_signature", ""),
                "by_source_length": eval_data.get("by_source_length", {
                    "short_1_10": {"n": 376, "bleu": 26.21, "chrf_pp": 43.66},
                    "medium_11_20": {"n": 449, "bleu": 25.19, "chrf_pp": 43.95},
                    "medium_21_40": {"n": 352, "bleu": 22.84, "chrf_pp": 42.25},
                }),
                "note": "Điểm số đo lường trên tập test đã lọc của project (IWSLT'15), không phản ánh phần trăm câu dịch đúng tuyệt đối.",
            },
            "disclaimer": "Demo học tập phục vụ nghiên cứu và hồ sơ thực tập; mô hình dịch các câu ngắn đến vừa và có thể gặp hạn chế về thuật ngữ chuyên ngành hoặc tên riêng."
        }

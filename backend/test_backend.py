"""
Kiểm thử tích hợp cho Backend EnVi Lab.
Sử dụng httpx.AsyncClient để kiểm tra:
1. /api/health
2. /api/model-info
3. /api/translate với model thật trên 3 câu mẫu
4. Các trường hợp biên (chuỗi rỗng, vượt giới hạn 40 từ)
"""

import asyncio
import sys
from pathlib import Path
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from main import app, model_service


async def run_tests():
    print("--- BẮT ĐẦU CHẠY KIỂM THỬ BACKEND VỚI MODEL THẬT ---")
    
    # Nạp model trực tiếp
    model_service.load_model()
    assert model_service.is_ready is True, f"Model chưa sẵn sàng, thiếu: {model_service.missing_files}"
    print("PASS: model_service.load_model() thành công.")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Health check
        res_health = await client.get("/api/health")
        assert res_health.status_code == 200, f"Health check thất bại: {res_health.text}"
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["model_ready"] is True
        print("PASS: test_health_check (status: ok, model_ready: True)")

        # 2. Model info
        res_info = await client.get("/api/model-info")
        assert res_info.status_code == 200
        info_data = res_info.json()
        assert info_data["limits"]["max_words"] == 40
        assert info_data["limits"]["max_tokens"] == 96
        assert info_data["evaluation"]["bleu"] == 24.14
        assert info_data["evaluation"]["chrf_pp"] == 43.05
        print(f"PASS: test_model_info (BLEU: {info_data['evaluation']['bleu']}, chrF++: {info_data['evaluation']['chrf_pp']})")

        # 3. Translation with real model
        samples = [
            ("I am a student.", "Tôi là một sinh viên."),
            ("I want to learn English and find a good job.", "Tôi muốn học tiếng Anh và tìm một công việc tốt."),
            ("This is an interesting project.", "Đây là một dự án thú vị.")
        ]
        for source, expected in samples:
            res_trans = await client.post("/api/translate", json={"text": source})
            assert res_trans.status_code == 200, f"Dịch thất bại cho: {source}, lỗi: {res_trans.text}"
            trans_data = res_trans.json()
            assert trans_data["translated_text"] == expected, f"Kết quả không khớp: {trans_data['translated_text']} != {expected}"
            assert trans_data["source_word_count"] > 0
            assert trans_data["source_token_count"] > 0
            assert trans_data["latency_ms"] > 0
            print(f"PASS: Dịch '{source}' -> '{trans_data['translated_text']}' ({trans_data['latency_ms']} ms)")

        # 4. Empty input
        res_empty = await client.post("/api/translate", json={"text": "   "})
        assert res_empty.status_code == 400
        assert "nhập" in res_empty.json()["detail"].lower()
        print("PASS: test_empty_input (HTTP 400 với thông báo chuẩn)")

        # 5. Word count limit (>40 words)
        long_sentence = " ".join(["word"] * 45) + "."
        res_long = await client.post("/api/translate", json={"text": long_sentence})
        assert res_long.status_code == 400
        assert "giới hạn" in res_long.json()["detail"].lower()
        print("PASS: test_exceed_max_words (HTTP 400 với thông báo chuẩn)")

    print("=== TẤT CẢ KIỂM THỬ BACKEND ĐỀU ĐẠT THÀNH CÔNG (100% PASS) ===")


if __name__ == "__main__":
    asyncio.run(run_tests())

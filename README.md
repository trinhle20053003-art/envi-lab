---
title: EnVi Lab - English to Vietnamese Machine Translation
emoji: 🌐
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# EnVi Lab — Website Dịch Máy Nơ-ron Anh – Việt (BiLSTM + Bahdanau Attention)

> **Portfolio Project:** Hệ thống Full-Stack Dịch máy Nơ-ron (Neural Machine Translation - NMT) từ Anh sang Việt chạy suy luận trực tiếp trên CPU cục bộ, phục vụ hồ sơ ứng tuyển Thực tập sinh AI Engineer / Full-Stack Developer.

---

## 1. Bài toán và Mục tiêu của Project

Project giải quyết bài toán dịch máy tự động cho cặp ngôn ngữ **Anh → Việt** trên ngữ liệu đàm thoại mở (IWSLT'15 TED Talks).

- **Kiến trúc mô hình:** Sequence-to-Sequence với bộ mã hóa BiLSTM 2 chiều (512 units tổng) kết hợp cơ chế tập trung cộng tính Bahdanau Attention (128 units) và bộ giải mã LSTM 1 chiều (512 units).
- **Mục tiêu sản phẩm:** Xây dựng một ứng dụng web hoàn chỉnh, phản hồi thời gian thực, nạp trọng số mô hình đã huấn luyện (`model.keras`) và SentencePiece tokenizer (`tokenizer_en.model`, `tokenizer_vi.model`) mà không phụ thuộc vào bất kỳ API dịch bên thứ ba nào.

---

## 2. Kiến trúc Ứng dụng & Cấu trúc Thư mục

```text
proud-nobel/
├── artifacts/
│   └── envi_tensorflow_model/       # Thư mục chứa gói model và báo cáo kỹ thuật gốc
│       ├── model.keras              # Trọng số mô hình Sequence-to-Sequence đã train
│       ├── tokenizer_en.model       # SentencePiece BPE model tiếng Anh (8.000 vocab)
│       ├── tokenizer_vi.model       # SentencePiece BPE model tiếng Việt (8.000 vocab)
│       ├── config.json              # Tham số mô hình (units, vocab_size, max_tokens)
│       ├── evaluation.json          # Điểm BLEU (24.14), chrF++ (43.05), thông số kiểm thử
│       ├── run_summary.json         # Tóm tắt quá trình huấn luyện và chọn checkpoint
│       ├── data_manifest.json       # Nguồn gốc tập dữ liệu IWSLT'15 và mã băm
│       ├── learning_curve.png       # Biểu đồ hàm mất mát loss/NLL qua 19 epoch
│       ├── inference_tf.py          # Module suy luận gốc
│       ├── model_tf.py              # Đăng ký custom layers (BahdanauAttention, PaddingMask,...)
│       └── data.py                  # Chuẩn hóa văn bản Unicode NFC và xử lý token
├── backend/
│   ├── venv/                        # Virtual environment Python riêng cho backend
│   ├── .env                         # Cấu hình cổng và đường dẫn model
│   ├── .env.example                 # Mẫu cấu hình môi trường
│   ├── main.py                      # FastAPI app, REST endpoints, lifespan management
│   ├── model_service.py             # Adapter nạp model 1 lần, CPU thread lock, đo độ trễ
│   ├── test_backend.py              # Kịch bản kiểm thử tích hợp model thật (100% pass)
│   └── requirements.txt             # Danh sách thư viện Python cần thiết
├── frontend/
│   ├── public/
│   │   └── learning_curve.png       # Biểu đồ huấn luyện hiển thị trên giao diện
│   ├── src/
│   │   ├── App.tsx                  # Giao diện chính EnVi Lab (bảng dịch, biểu đồ, lịch sử)
│   │   ├── App.css                  # Bố cục responsive, bảng màu Navy/Teal hiện đại
│   │   ├── index.css                # CSS variables, typography, reset
│   │   ├── types.ts                 # TypeScript interfaces cho API và trạng thái
│   │   └── main.tsx                 # Điểm khởi chạy React 18
│   ├── index.html                   # Khai báo font Plus Jakarta Sans & JetBrains Mono
│   ├── package.json                 # Cấu hình dependency frontend (React, Vite, TypeScript)
│   └── vite.config.ts               # Proxy cấu hình `/api` -> `http://127.0.0.1:8008`
├── .env.example                     # Mẫu biến môi trường gốc
└── README.md                        # Hướng dẫn chi tiết sử dụng và báo cáo kỹ thuật
```

---

## 3. Vị trí Đặt Tệp Model và Tokenizer

Để ứng dụng tự động nhận diện mô hình, hãy đảm bảo các tệp nằm đúng trong thư mục `artifacts/envi_tensorflow_model`:

| Tên tệp | Vai trò |
| :--- | :--- |
| `model.keras` | Trọng số mạng nơ-ron Keras (9.171.136 tham số) |
| `tokenizer_en.model` | Mô hình phân tách từ con SentencePiece BPE tiếng Anh |
| `tokenizer_vi.model` | Mô hình phân tách từ con SentencePiece BPE tiếng Việt |
| `config.json` | Cấu hình giới hạn từ (`max_words: 40`) và token (`max_tokens: 96`) |
| `evaluation.json` | Báo cáo điểm số BLEU và chrF++ thực tế |
| `learning_curve.png` | Ảnh biểu đồ mất mát huấn luyện |

> **Lưu ý:** Bạn có thể tùy biến đường dẫn này thông qua biến môi trường `MODEL_DIR` trong tệp `backend/.env`. Nếu thiếu bất kỳ tệp nào, backend sẽ thông báo rõ ràng danh sách tệp thiếu qua API `/api/health` thay vì tạo lỗi sập server.

---

## 4. Yêu cầu Môi trường & Cài đặt

- **Hệ điều hành:** Windows 10/11, macOS hoặc Linux.
- **Python:** Khuyến nghị Python 3.10 – 3.13 (đã thử nghiệm thực tế thành công trên Python 3.13.13).
- **Node.js:** Phiên bản 18+ hoặc 20+ (đã thử nghiệm trên v24.15.0).

### Bước 4.1: Cài đặt Backend (Python)

1. Mở PowerShell hoặc Terminal tại thư mục gốc của project:
   ```powershell
   cd backend
   ```
2. Tạo môi trường ảo:
   ```powershell
   python -m venv venv
   ```
3. Kích hoạt môi trường và cài đặt các gói phụ thuộc:
   - **Trên Windows PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
   - **Trên macOS / Linux:**
     ```bash
     source venv/bin/activate
     pip install -r requirements.txt
     ```

### Bước 4.2: Cài đặt Frontend (React + Vite)

1. Mở một cửa sổ Terminal khác và di chuyển vào thư mục `frontend`:
   ```powershell
   cd frontend
   npm install
   ```

---

## 5. Lệnh Chạy Ứng Dụng Local

### Khởi chạy Backend (Cổng 8008)

Tại thư mục `backend` (đã kích hoạt venv):
```powershell
# Windows PowerShell:
$env:KERAS_HOME = "$pwd\.keras"
python -m uvicorn main:app --host 127.0.0.1 --port 8008 --reload
```

*Hoặc trên macOS / Linux:*
```bash
export KERAS_HOME="$(pwd)/.keras"
python -m uvicorn main:app --host 127.0.0.1 --port 8008 --reload
```

Backend sẽ nạp model một lần duy nhất vào bộ nhớ khi khởi động, chạy warm-up suy luận 1 câu và sẵn sàng tại:  
👉 **API Base URL:** `http://127.0.0.1:8008`  
👉 **Tài liệu Swagger UI:** `http://127.0.0.1:8008/docs`

### Khởi chạy Frontend (Cổng 5173)

Tại thư mục `frontend`:
```powershell
npm run dev
```

Vite Development Server sẽ khởi chạy tại:  
👉 **Địa chỉ website:** `http://localhost:5173`

*(Mọi yêu cầu gọi đến `/api/...` từ giao diện React sẽ tự động được Vite proxy sang backend `http://127.0.0.1:8008` mà không gặp lỗi CORS hay hardcode domain).*

---

## 6. Danh sách REST API Chính

| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Kiểm tra trạng thái máy chủ, model đã sẵn sàng chưa, và liệt kê các tệp thiếu nếu có. |
| `POST` | `/api/translate` | Tiếp nhận `{ "text": "I am a student." }`, trả về câu dịch tiếng Việt, số từ, số token và thời gian suy luận (`latency_ms`). |
| `GET` | `/api/model-info` | Cung cấp thông số kỹ thuật (kiến trúc BiLSTM, attention units, thông số tập test, BLEU, chrF++). |

---

## 7. Kết quả Thực nghiệm & Kiểm thử

### Thông số Đánh giá Mô hình Thực tế (Đo lường trên tập Test IWSLT'15 đã lọc)

- **Test BLEU:** `24.14` (SacreBLEU 2.6.0, 13a tokenizer).
- **Test chrF++:** `43.05` (Phản ánh độ chính xác mức ký tự và mẩu từ).
- **Tổng số tham số:** `9.171.136`.
- **Độ trễ suy luận trên CPU:** ~40ms – 65ms cho câu ngắn và vừa.

| Nhóm câu theo số từ nguồn | Số câu test (n) | Điểm BLEU | Điểm chrF++ |
| :--- | :---: | :---: | :---: |
| Câu ngắn (1 – 10 từ) | 376 | **26.21** | **43.66** |
| Câu vừa (11 – 20 từ) | 449 | **25.19** | **43.95** |
| Câu dài (21 – 40 từ) | 352 | **22.84** | **42.25** |

> **Lưu ý đánh giá:** Điểm số trên được tính theo quy chuẩn học thuật của bài toán dịch máy trên tập test đã lọc của project, **không phải là phần trăm câu dịch đúng tuyệt đối**.

### Chạy Kịch bản Kiểm thử Tự động Backend

Backend đi kèm bộ kiểm thử tự động toàn diện kiểm tra trên chính model thật:
```powershell
python backend/test_backend.py
```
Bộ test bao gồm:
1. `test_health_check`: Đảm bảo API trả mã 200 và `model_ready = True`.
2. `test_model_info`: Kiểm tra tính toàn vẹn của siêu dữ liệu kiến trúc.
3. `test_translation_real_model`: Đối chiếu câu dịch thực tế từ model với các câu mẫu chuẩn.
4. `test_empty_input`: Kiểm tra từ chối chuỗi rỗng / khoảng trắng (trả lỗi 400 rõ ràng).
5. `test_exceed_max_words`: Kiểm tra từ chối câu vượt quá 40 từ (trả lỗi 400 thân thiện).

---

## 8. Đoạn Mô tả Project Tham khảo cho CV

Bạn có thể đưa đoạn văn ngắn gọn, đúng trọng tâm này vào mục **Projects** trong CV của mình:

> **English-Vietnamese Neural Machine Translation Web Application (EnVi Lab)**  
> *Công nghệ: Python, TensorFlow 2.21, Keras 3, SentencePiece, FastAPI, React, TypeScript, Vite.*  
> - Xây dựng hệ thống dịch máy nơ-ron Anh – Việt end-to-end từ tiền xử lý ngữ liệu IWSLT'15 (115.154 cặp câu), tối ưu pipeline phân tích từ con SentencePiece BPE (8k từ vựng).  
> - Triển khai mô hình Sequence-to-Sequence (BiLSTM Encoder 512 units + Additive Bahdanau Attention 128 units + LSTM Decoder 512 units, 9.17M tham số) đạt 24.14 BLEU và 43.05 chrF++ trên tập kiểm thử chuẩn hóa.  
> - Thiết kế kiến trúc backend REST API với FastAPI quản lý vòng đời (lifespan) nạp model 1 lần, điều phối suy luận an toàn đa luồng trên CPU với độ trễ ~45–65ms.  
> - Xây dựng giao diện web phản hồi nhanh bằng React & TypeScript, hỗ trợ phím tắt, đếm từ/token thời gian thực, lưu lịch sử phiên dịch và trực quan hóa biểu đồ học tập (learning curve).

---

## 9. Hướng dẫn Triển khai Lên Web Miễn phí (Hugging Face Spaces)

Dự án đã được cấu hình sẵn `Dockerfile` chuẩn Docker Space cho **Hugging Face Spaces** (miễn phí 16GB RAM, 2 vCPU, 50GB ổ cứng).

### Các bước triển khai:

1. **Tạo Space mới trên Hugging Face:**
   - Truy cập: [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Đặt tên Space: `envi-lab` (hoặc tên tùy thích).
   - License: chọn `mit` hoặc `apache-2.0`.
   - Chọn **Space SDK:** Chọn **Docker** (Blank).
   - Chọn **Space hardware:** **Free (2 vCPU, 16 GB RAM)**.
   - Nhấn **Create Space**.

2. **Đẩy mã nguồn lên Space:**
   Tại thư mục gốc của project trên máy tính của bạn, mở terminal/PowerShell:
   ```bash
   # Khởi tạo git nếu chưa có
   git init
   git branch -M main

   # Thêm toàn bộ mã nguồn, model và cấu hình Docker
   git add .
   git commit -m "Deploy EnVi Lab Full-stack to Hugging Face Spaces"

   # Kết nối remote tới Space của bạn (thay <your-username> bằng tên tài khoản Hugging Face của bạn)
   git remote add space https://huggingface.co/spaces/<your-username>/envi-lab

   # Đẩy code lên (nhập Hugging Face Access Token với quyền Write khi được hỏi mật khẩu)
   git push space main --force
   ```

3. **Truy cập ứng dụng:**
   - Hugging Face Spaces sẽ tự động đọc `Dockerfile`, build container và chạy ứng dụng trên cổng `7860`.
   - Sau 2-3 phút, website của bạn sẽ hoạt động trực tiếp tại địa chỉ public:
     `https://huggingface.co/spaces/<your-username>/envi-lab`
   - Đường dẫn này có thể gắn trực tiếp vào CV hoặc hồ sơ LinkedIn/GitHub để nhà tuyển dụng trải nghiệm ngay lập tức.


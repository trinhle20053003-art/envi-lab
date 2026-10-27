"""
FastAPI Backend cho ứng dụng EnVi Lab (Machine Translation Anh -> Việt).
Cung cấp REST API:
- GET  /api/health: Kiểm tra trạng thái máy chủ và model
- POST /api/translate: Thực hiện dịch câu tiếng Anh sang tiếng Việt
- GET  /api/model-info: Thông số kiến trúc và kết quả thực nghiệm
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_service import ModelService

# Nạp biến môi trường từ .env
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(name)s: %(message)s"
)
logger = logging.getLogger("envi_backend")

# Khởi tạo singleton model service
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: nạp model một lần khi khởi động."""
    logger.info("Khởi động FastAPI server EnVi Lab...")
    model_service.load_model()
    yield
    logger.info("Dừng FastAPI server EnVi Lab.")


app = FastAPI(
    title="EnVi Lab API",
    description="Backend API phục vụ dịch máy Anh - Việt dựa trên mô hình BiLSTM + Bahdanau Attention",
    version="1.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS để frontend React kết nối thuận tiện khi phát triển
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        description="Câu tiếng Anh cần dịch (tối đa 40 từ và 96 subwords)",
        examples=["I am a student."]
    )


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    latency_ms: float
    latency_sec: float
    source_word_count: int
    source_token_count: int


@app.get("/api/health")
def check_health():
    """Kiểm tra trạng thái sẵn sàng của backend và model."""
    if model_service.is_ready:
        return {
            "status": "ok",
            "model_ready": True,
            "message": "Hệ thống EnVi Lab và model dịch đã sẵn sàng.",
            "missing_files": []
        }
    else:
        return {
            "status": "degraded",
            "model_ready": False,
            "message": "Backend đang chạy nhưng model chưa sẵn sàng.",
            "missing_files": model_service.missing_files
        }


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Dịch một câu tiếng Anh sang tiếng Việt."""
    if not model_service.is_ready:
        missing = f" (Tệp thiếu: {', '.join(model_service.missing_files)})" if model_service.missing_files else ""
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Mô hình dịch hiện chưa sẵn sàng{missing}. Vui lòng kiểm tra lại tệp model."
        )

    try:
        result = await model_service.translate_text(request.text)
        return result
    except ValueError as val_err:
        # Lỗi dữ liệu người dùng (chuỗi rỗng, vượt quá 40 từ, vượt token)
        logger.warning(f"Lỗi xác thực đầu vào: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except RuntimeError as run_err:
        logger.error(f"Lỗi runtime: {run_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(run_err)
        )
    except Exception as e:
        logger.exception("Lỗi không mong muốn trong quá trình suy luận:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Đã xảy ra lỗi nội bộ trong quá trình xử lý câu dịch."
        )


@app.get("/api/model-info")
def get_model_info():
    """Lấy thông tin kiến trúc, cấu hình và kết quả thực nghiệm của model."""
    return model_service.get_info()


# Phục vụ giao diện tĩnh Frontend (React build)
dist_candidates = [
    Path(__file__).parent.parent / "frontend" / "dist",
    Path(__file__).parent / "dist",
    Path("/home/user/app/frontend/dist"),
    Path("/app/frontend/dist"),
]
dist_dir: Optional[Path] = None
for candidate in dist_candidates:
    if candidate.exists() and (candidate / "index.html").exists():
        dist_dir = candidate.resolve()
        break

if dist_dir:
    logger.info(f"Đã kích hoạt phục vụ giao diện Frontend tĩnh từ: {dist_dir}")
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Endpoint không tồn tại")
        file_path = dist_dir / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_dir / "index.html")
else:
    @app.get("/")
    def read_root():
        return {
            "app": "EnVi Lab API",
            "description": "NMT English -> Vietnamese Translation Service",
            "model_ready": model_service.is_ready,
            "docs_url": "/docs"
        }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8008"))
    uvicorn.run("main:app", host=host, port=port, reload=False, workers=1)

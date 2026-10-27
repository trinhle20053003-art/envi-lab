# Dùng model đã huấn luyện

Giải nén ZIP, mở terminal trong thư mục envi_tensorflow_model. Tạo môi trường Python theo run_summary.json rồi cài:

    pip install -r requirements.txt

Dịch trong Python:

    from inference_tf import load_translator
    translator = load_translator('.')
    print(translator.translate('I want to learn English and find a good job.'))

model.keras đã huấn luyện; không cần chạy lại notebook để dùng. CPU có thể dùng để suy luận. Trang web sẽ cần gọi load_translator một lần ở backend. Không tải model cho mỗi yêu cầu. Model vẫn có lỗi nghĩa và tên riêng; cần nêu rõ là demo học tập.

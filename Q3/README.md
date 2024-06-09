# Các bước chuẩn bị notebook trên Kaggle

## 1. Tạo notebook trên Kaggle

- Truy cập vào trang chủ của [Kaggle](https://www.kaggle.com/).
- Nhấn nút `Create` bên trái, chọn `New Notebook`.
- Kaggle sẽ chuyển hướng sang một notebook trống. Hãy vào `File` -> `Import Notebook` và upload file notebook trong thư mục `Q3` của repo này.

## 2. Import dataset của cuộc thi

Dataset được thêm tự động vào notebook khi import notebook. Nếu không tự động, hãy thêm dataset bằng cách:
- Nhấn nút `Add Input Data` ở bên phải.
- Tại khung tìm kiếm, gõ `Learning Agency Lab - Automated Essay Scoring 2.0`. Sau đó nhấn dấu cộng `+` để thêm dataset vào notebook.

## 3. Chuẩn bị token API của Hugging Face

- Truy cập vào repo [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) để xin quyền truy cập vào model.
- Sau khi đã có quyền truy cập, tiếp tục vào trang [Access Tokens](https://huggingface.co/settings/tokens) để tạo token API và lưu lại.
- Quay lại notebook, nhấn nút `Add-ons` -> `Secrets` -> `Add a new secret`. Đặt tên secret là `HF_TOKEN`, giá trị là token vừa tạo.

Trong trường hợp thời gian phê duyệt quá lâu, hãy sử dụng token sau
```
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
```
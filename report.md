## 1. EDA (Exploratory Data Analysis)
## 2. Transformer models dùng cho xử lý lý ngôn ngữ tự nhiên (https://huggingface.co/learn/nlp-course/chapter1/1)
### 2.1. Định nghĩa, các thức hoạt động
### 2.2. Phân loại (3 loại)
### 2.3. Cấu trúc pipelines (chapter  2)	
## 3. Fine tuning mô hình DeBERTaV3
### 3.1. Giới thiệu mô hình 
### 3.2. Tiền xử lý dữ liệu và tokenizer (các hàm preprocessing)
### 3.3. Các tham số quan trọng

**Mô hình DeBERTaV3**
- Head `ForSequenceClassification`: Thêm layer classifier cho mô hình gốc.
- `num_labels`: Số lượng labels mong muốn mô hình có thể dự đoán. Trong bài này là 6, vì điểm số của essay là số nguyên trải dài từ 1 đến 6. Nguồn [Link to the Holistic Scoring Rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf).

**Tokenizer**:
- `truncation`: Điều khiển độ dài tối đa nếu là `True`, khi đó độ dài tối đa được quy định bởi `max_length`.
- `max_length`: Độ dài tối đa được dùng bởi `truncation`. Trong bài này được điều chỉnh thành 1024.

**TrainingArguments**:
- `learning_rate`: Hệ số học.
- `warmup_ratio`: Tỉ lệ warmup.
- `num_train_epochs`: Số lượng epochs.
- `per_device_train_batch_size`: Kích thước batch huấn luyện trên mỗi thiết bị (CPU, GPU)
- `per_device_eval_batch_size`: Kích thước batch đánh giá trên mỗi thiết bị.
- `fp16`: Sử dụng floating point 16 để tăng tốc quá trình huấn luyện.
- `lr_scheduler_type`: Loại learning rate scheduler áp dụng cho quá trình training.
- `weight_decay`: Hệ số weight decay áp dụng cho tất cả các tầng từ bias và LayerNorm.

### 3.4. Train mô hình sử dụng StratifiedKFold

Dataset được chia thành các fold với tỉ lệ label cân bằng nhờ `StratifiedKFold`, các fold này được huấn luyện trên cùng một mô hình. Sau khi kết thúc quá trình cho một fold, mô hình được lưu lại để thử nghiệm và gửi kết quả.

### 3.5. Điều chỉnh mô hình về dự đoán regression
## 4. Prompt Engineering với Meta-Llama-3-8B-Instruct
### 4.1. Giới thiệu mô hình
### 4.2. Các bước thực hiện

**Khởi tạo pipeline**:

Loại pipeline được chọn là `text-generation` và `device_map=auto` để hệ thống quyết định thiết bị nào sẽ được sử dụng.

**Prompt Engineering**

*Input: Đoạn văn cần chấm diểm.*

*Output: Prompt chấm điểm sau khi áp dụng chat template.*

    messages = [
        {'role': 'system', 'content': 'You are a strict teacher.'},
        {'role': 'user', 'content': instruction + essay},
        {'role': 'assistant', 'content': f'\n\nThe score is: '}
    ]

Messages gồm các role:
- `system`: Cho biết ngữ cảnh và chỉ dẫn ban đầu của prompt.
- `user`: Yêu cầu đặt ra cần được giải quyết. Ở đây yêu cầu chấm điểm từ 1-6 và kèm theo đoạn văn.
- `assistant`: Phản hồi mà người dùng mong muốn.

Sau đó trả về messages.

**Scoring**

*Input: Messages từ bước trên*

*Output: Điểm cho những essay đó*

Messages được đưa vào pipeline, điểm được trích xuất và trả về cho hàm.

**Lặp lại cho cả dataset**

Dùng method `map` của HuggingFace Dataset để thực hiện song song hóa việc tính điểm để tận dụng tài nguyên và tăng tốc độ xử lý.
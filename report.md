## 1. EDA (Exploratory Data Analysis)
## 2. Transformer models dùng cho xử lý lý ngôn ngữ tự nhiên (https://huggingface.co/learn/nlp-course/chapter1/1)
### 2.1. Định nghĩa, các thức hoạt động
### 2.2. Phân loại (3 loại)
### 2.3. Cấu trúc pipelines (chapter  2)	
## 3. Fine tuning mô hình DeBERTaV3
### 3.1. Giới thiệu mô hình 
#### DeBERTaV3: Cải thiện DeBERTa bằng cách sử dụng Tiền huấn luyện kiểu ELECTRA với Chia sẻ Nhúng Gradient-Disentangled

- DeBERTa cải thiện các mô hình BERT và RoBERTa bằng cách sử dụng cơ chế tập trung disentangled và bộ giải mã mặt nạ cải tiến. Với hai cải tiến đó, DeBERTa vượt trội hơn RoBERTa trong hầu hết các tác vụ hiểu ngôn ngữ tự nhiên (NLU) với 80GB dữ liệu huấn luyện.

- Trong DeBERTa V3, các tác giả đã cải thiện hiệu quả của DeBERTa bằng cách sử dụng tiền huấn luyện kiểu ELECTRA với Chia sẻ Nhúng Gradient-Disentangled. So với DeBERTa, phiên bản V3 của các tác giả cải thiện đáng kể hiệu suất mô hình trên các tác vụ downstream. Bạn có thể tìm hiểu thêm chi tiết kỹ thuật về mô hình mới từ bài báo của chúng tôi.

- Mô hình cơ sở DeBERTa V3 đi kèm với 12 lớp và kích thước ẩn là 768. Nó chỉ có 86M tham số backbone với từ điển chứa 128K token, giới thiệu 98M tham số trong lớp Nhúng. Mô hình này được huấn luyện bằng 160GB dữ liệu như DeBERTa V2.

Nếu cần biết thêm chi tiết triển khai và cập nhật thì hãy truy cập [repo](https://github.com/microsoft/DeBERTa?tab=readme-ov-file) này.

#### Một số ứng dụng nổi bật:
1. Tạo Nội Dung Tự Động: DeBERTa v3 có thể tạo ra các bài đăng blog, bài viết tin tức, và mô tả sản phẩm chất lượng cao cho các ứng dụng khác nhau. Tính năng này có thể rất hữu ích cho các doanh nghiệp cần tạo ra lượng lớn nội dung nhanh chóng.

2. Chatbots và Trợ lý Cá Nhân: DeBERTa v3 là lựa chọn tuyệt vời cho chatbots và trợ lý cá nhân do khả năng tạo ra các phản hồi có ý nghĩa và mạch lạc. Mô hình có thể được sử dụng bởi các ứng dụng này để hiểu các truy vấn của người dùng và tạo ra các phản hồi phù hợp.

3. Phân Tích Cảm Xúc: Các doanh nghiệp cần theo dõi cảm xúc về thương hiệu trên các nền tảng truyền thông xã hội hoặc các trang web đánh giá có thể thấy mức độ chính xác cao của DeBERTa v3 trong việc phân tích cảm xúc rất hữu ích. Mô hình có khả năng phân tích nhanh chóng lượng lớn văn bản và tiết lộ ý kiến và sở thích của khách hàng.

4. Trả lời Câu Hỏi: Hiệu suất ấn tượng của DeBERTa v3 trong các tác vụ trả lời câu hỏi có thể được sử dụng trong nhiều ứng dụng, bao gồm dịch vụ khách hàng, chatbots, và công cụ tìm kiếm. Mô hình có thể nhanh chóng hiểu các truy vấn của người dùng và cung cấp các phản hồi chính xác dựa trên dữ liệu có sẵn.

#### Fine-tuning trên các tác vụ NLU
| Model | Vocabulary(K) | Backbone #Params(M) | SQuAD 2.0(F1/EM) | MNLI-m/mm(ACC) |
|-------|---------------|---------------------|------------------|----------------|
| RoBERTa-base | 50 | 86 | 83.7/80.5 | 87.6/- |
| XLNet-base | 32 | 92 | -/80.2 | 86.8/- |
| ELECTRA-base | 30 | 86 | -/80.5 | 88.8/ |
| DeBERTa-base | 50 | 100 | 86.2/83.1 | 88.8/88.5 |
| DeBERTa-v3-base | 128 | 86 | 88.4/85.4 | 90.6/90.7 |
| DeBERTa-v3-base + SiFT | 128 | 86 | -/- | 91.0/- |

#### Fine-tuning với HF transformers

```bash
#!/bin/bash

cd transformers/examples/pytorch/text-classification/

pip install datasets
export TASK_NAME=mnli

output_dir="ds_results"

num_gpus=8

batch_size=8

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 256 \
  --warmup_steps 500 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir $output_dir

```

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
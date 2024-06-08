## 1. EDA (Exploratory Data Analysis)
## 2. Transformer models dùng cho xử lý lý ngôn ngữ tự nhiên (https://huggingface.co/learn/nlp-course/chapter1/1)
### 2.1. Định nghĩa, các thức hoạt động
### 2.2. Phân loại (3 loại)
### 2.3. Cấu trúc pipelines (chapter  2)	
## 3. Fine tuning mô hình DeBERTaV3
### 3.1. Giới thiệu mô hình 
### 3.2. Tiền xử lý dữ liệu và tokenizer (các hàm preprocessing)
### 3.2.1. Tokenizer của mô hình
- Tokenizer của mô hình sẽ chuyển văn bản thành một dãy các token. Dãy token này sẽ có 3 thành phần để biểu diễn:
    - `input_ids`: Các token đã được mã hóa thành dãy các số nguyên ứng với từ điển của mô hình.
    - `attention_mask`: Một dãy số 0 và 1, 1 ở vị trí token, 0 ở vị trí padding (các token thêm vào để đảm bảo độ dài của dãy token bằng nhau).
    - `token_type_ids`: Dùng để phân biệt các token trong nhiều input khác nhau. Ở bài toán này, chỉ có một input nên giá trị của `token_type_ids` tất cả là 0.
- Để sử dụng tokenizer của mô hình, ta sử dụng `AutoTokenizer` từ `transformers` để load từ checkpoint đã được tải sẵn:
    ```python
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
    ```
- Sau khi đã load tokenizer, ta sẽ sử dụng hàm `tokenizer` để mã hóa văn bản thành dãy token:
    ```python
    tokenizer(x['full_text'], truncation = True, max_length = max_length)
    ```
- Trong đó:
    - `x['full_text']`: Văn bản cần mã hóa.
    - `truncation`: Cắt bớt văn bản nếu vượt quá `max_length`.
    - `max_length`: Độ dài tối đa của dãy token.
- Một ví dụ về kết quả trả về của hàm `tokenizer`:
    ```python
    text = "Many people have car where they live."
    ```
    - Sử dụng hàm `tokenizer.tokenize(text)` ta có thể thấy được các token được tạo ra như sau (dấu `▁` ở đầu mỗi token là ký hiệu của một từ):
    ```python
    ['▁Many', '▁people', '▁have', '▁car', '▁where', '▁they', '▁live', '.']
    ```
    - Sử dụng hàm `tokenizer(text, truncation = True, max_length = 20)` ta có thể thấy được kết quả trả về:
    ```python
    {'input_ids': [1, 1304, 355, 286, 640, 399, 306, 685, 260, 2], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    ```
- Có một vài điểm chú ý:
    - Ở đây chỉ có một input nên max_length sẽ là độ dài của input.
    - Chỉ có 8 token nhưng có 10 phần tử trong `input_ids`, `token_type_ids` và `attention_mask` vì có thêm token `[CLS]` và `[SEP]` ở đầu và cuối dãy token.
    - `token_type_ids` chỉ có giá trị 0 vì chỉ có một input.
    - `attention_mask` chỉ có giá trị 1 vì không có token nào là padding.
### 3.2.2. Tiền xử lý dữ liệu
Trong các bài tiểu luận xuất hiện một vài các lỗi có thể gây ảnh hưởng quá trình tokenization nên cần phải xử lý trước khi mã hóa văn bản thành dãy token. Các lỗi đó bao gồm:
- Xuất hiện chuối `\xa0` (non-breaking space).
- Xuất hiện chuỗi các chuỗi html do web scraping.
- Xuất hiện các dấu chấm liên tiếp.
- Xuất hiện các dấu phẩy liên tiếp.
- Xuất hiện chuỗi `''` thay vì `"`
Ta sẽ sử dụng hàm `preprocessor` để xử lý các lỗi trên
### 3.2.3. Hàm tiền xử lý dữ liệu cuối cùng:
```python
def data_preprocessing(path, tokenizer, max_length):
    data = pd.read_csv(path)
    data['label'] = data['score'].map(lambda x: x - 1)
    data["label"] = data["label"].astype("float32")
    data['full_text'] = data['full_text'].apply(preprocessor)
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(lambda x: tokenizer(x['full_text'], truncation = True, max_length = max_length), batched = True)
    columns_to_remove = ['essay_id', 'full_text', 'score']
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset
```
- Hàm `data_preprocessing` nhận vào đường dẫn của file dữ liệu, tokenizer và max_length.
- Hàm đọc file dữ liệu, chuẩn hóa cột `score` thành cột `label` để phù hợp với mô hình.
- Loại bỏ các lỗi trong cột `full_text` bằng hàm `preprocessor`.
- Sử dụng đối tượng `Dataset` để lưu trữ dữ liệu.
- Sử dụng hàm `map` để mã hóa văn bản thành dãy token.
    - `truncation = True` để cắt bớt văn bản nếu vượt quá `max_length`.
    - `max_length` là độ dài tối đa của dãy token được định nghĩa là 1024. Vì có khá ít văn bản có độ dài lớn hơn 1024 nên ta sẽ chọn cắt các văn bản dài.
    - `batched = True` để mã hóa dãy token có thể được thực hiện song song trên nhiều văn bản.
- Vì mô hình chỉ cần biết về cấu trúc của dãy token nên ta sẽ loại bỏ các cột không cần thiết còn lại
### 3.3. Các tham số quan trọng
### 3.4. Train mô hình sử dụng StratifiedKFold
### 3.5. Điều chỉnh mô hình về dự đoán regression
Có 2 hướng để có thể sử dụng mô hình DeBERTaV3 để dự đoán 6 điểm số của bài luận.
- Hướng classification: Các điểm số sẽ được mã hóa về các vector sau để đảm bảo tính thứ tự:
    - 1: [1, 0, 0, 0, 0, 0]
    - 2: [1, 1, 0, 0, 0, 0]
    - 3: [1, 1, 1, 0, 0, 0]
    - 4: [1, 1, 1, 1, 0, 0]
    - 5: [1, 1, 1, 1, 1, 0]
    - 6: [1, 1, 1, 1, 1, 1]
- Hướng regression: bởi vì các điểm số này có tính thứ tự tự nhiên (1 < 2 < 3, ...) nên ta có thể dùng bài toán regression để dự đoán.

Sau quá trình thử nghiệm thì nhóm thấy hướng regression cho kết quả tốt hơn nên bài làm sẽ đi theo hướng này.

Để cho mô hình có thể dự đoán regression ta cần một vài điều chỉnh đặt biệt:
```python
model = AutoModelForSequenceClassification. \
                from_pretrained(cfg.checkpoint,
                num_labels=1,
                hidden_dropout_prob = 0,
                attention_probs_dropout_prob = 0,
                cache_dir='./cache')
```
- Đầu tiên ta điều chỉnh `num_labels=1` để mô hình chỉ dự đoán một giá trị điểm số duy nhất (ở đây là số thực)
- Tiếp theo ta sẽ tắt dropout cho mô hình. Dropout là một kỹ thuật regularization được sử dụng để ngăn chặn overfitting trong mạng neural bằng cách ngẫu nhiên "dropout" một phần của các đơn vị (neurons) trong quá trình huấn luyện. Trong trường hợp này, khi sử dụng regression với NLP transformers, việc loại bỏ dropout là cần thiết để tránh batch normalization gây ra sự không ổn định trong dự đoán của mô hình. Điều này là do dropout có thể làm giảm tính ổn định của batch normalization đối với việc dự đoán giá trị liên tục.
## 4. Prompt Engineering với Meta-Llama-3-8B-Instruct
### 4.1. Giới thiệu mô hình
### 4.2. Các bước thực hiện

## 1. EDA (Exploratory Data Analysis)
### 1.1 Giới thiệu bộ dữ liệu 
- Dữ liệu được chia ra 2 tập là tập dữ liệu **train** và dữ liệu **test** được tham khảo từ bộ dữ liệu cuộc thi bao gồm khoảng 24000 bài luận, tranh luận do học sinh viết [Đường dẫn đến the Holistic Scoring Rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf). Mỗi bài luận được chấm theo thang điểm từ 1 đến 6. Mục tiêu sẽ là từ bộ dữ liệu train xây dựng mô hình phù hợp để dự đoán số điểm mà một bài luận nhận được từ văn bản của nó (tập test).
### 1.2 Kiểm tra dữ liệu 
#### Kích thước tập dữ liệu 

|        | Train|Test|
|--------|------|----|
| Số dòng|17307 |3   |
| Số cột |  3   |2   |

#### Ý nghĩa của các dòng và cột 
Gồm 3 bộ dữ liệu
- train.csv: Các bài luận và điểm số được sử dụng làm dữ liệu train

| Field     | Description                             |
|-----------|-----------------------------------------|
| `essay_id`  | ID duy nhất của bài luận                  |
| `full_text` | Câu trả lời đầy đủ của bài luận                 |
| `score`   | Điểm tổng thể của bài luận theo thang điểm 1-6 |

- test.csv: Các bài luận được sử dụng làm dữ liệu test. Chứa các trường giống như train.csv, ngoại trừ trường score.

| Field     | Description                             |
|-----------|-----------------------------------------|
| `essay_id`  | ID duy nhất của bài luận              |
| `full_text` | Câu trả lời đầy đủ của bài luận       |    
#### Kiểu dữ liệu mỗi cột 
- Cột `essay_id` và cột `full_text` ở dữ liệu train và text đều có kiểu là **object**.
- Cột `score` ở bộ dữ liệu train là **int**.
#### Kiểm tra missing value  
- Dữ liệu không bị missing.
#### Kiểm tra trùng lắp
- Dữ liệu không bị trùng lắp.
### 2. Phân phối và thống kê dữ liệu 
#### Dữ liệu numerical 
- Cột dữ liệu có dạng numerical là cột `score` ở bộ dữ liệu train.
  - Bảng tóm tắt số liệu thống kê mô tả.
      |      | Score                             |
      |-----------|----------|
      | min | 1          |
      | lower_quartile | 2 |    
      | median | 3   |  
      | upper_quartile | 4  |   
      | max | 6  |    
  - Phân bố các bài luận theo điểm số.
  <figure>
      <img title="a title" alt="Alt text" src="rpimg/score_distribution.svg">
      <figcaption style="text-align: center;"><em>Phân bố các bài luận theo điểm số</em></figcaption>
  </figure>

##### Nhận xét:
- Phần lớn bài luận nhận được điểm số 3 (6280 bài), cho thấy đây là mức điểm phổ biến nhất.
Điểm số 2 và 4 cũng có số lượng bài luận khá cao, lần lượt là 4723 và 3926 bài.
- Điểm số 1 và 5 có số lượng bài luận ít hơn đáng kể, lần lượt là 1252 và 970 bài.
Điểm số 6 là mức điểm hiếm gặp nhất với chỉ 156 bài luận, cho thấy rất ít bài đạt được điểm số này.
- Có xu hướng giảm dần số lượng bài luận từ điểm số 3 xuống điểm số 6, cho thấy việc đạt được điểm số cao (5 và 6) là khó khăn hơn so với điểm số thấp hoặc trung bình (1-4).

#### Dữ liệu categorical 
- Do các cột còn lại là các cột mang ý nghĩa định danh và text, nên ta không thu được nhiều ý nghĩa qua cách phân tích trực tiếp .Nhóm sẽ khai thác, khám phá cột `full_text` và mối quan hệ giữa chúng với điểm số qua các câu hỏi phía dưới.
### 3. Các câu hỏi khám phá dữ liệu 
#### Phân bố của điểm số các bài luận theo độ dài

<figure>
      <img title="a title" alt="Alt text" src="rpimg/score_distribution_by_length.png">
      <figcaption style="text-align: center;"><em>Phân bố score theo độ dài văn bản</em></figcaption>
 </figure>
<figure>
      <img title="a title" alt="Alt text" src="rpimg/length_distribution.png">
      <figcaption style="text-align: center;"><em>Phân bố score theo độ dài văn bản</em></figcaption>
 </figure>

##### Nhận xét chung:
  - Những bài luận có độ dài càng lớn thì số điểm thấp càng ít.
  - Ở các bài luận (0-500] từ chỉ có 0.00697% được điểm 6 dù số lượng bài luận trong khoảng này rất lớn, trong khi (1000-1500] từ thì chiếm 27% điểm 6.
  - Hầu như các luận trên 1000 từ không có điểm 1.
  - Các luận (500-1000] rất đa dạng điểm, số lượng bài điểm 4, 5 chiếm đa số.
  - Các luận (1000-1500] phân bố điểm khá đều và không có điểm 1 cùng với số lượng khá ít cho thấy chất lượng có thể được đo đạc theo lượng từ trong bài.
=> Có thể thấy phần lớn bài luận có số lượng từ vựng nhiều sẽ có điểm số cao hơn.

  - Tuy nhiên khoảng (1500-2000] từ chỉ có một bài đạt điểm 2 không thể hiện được phán đoán gì. Ta xem thử bài văn này có nội dung như thế nào.

#### Kiểm tra các bài luận có chứa từ vựng sai chính tả, liệu điều này có ảnh hưởng đến điểm số không
<figure>
      <img title="a title" alt="Alt text" src="rpimg/misspelled_count_score.svg">
      <figcaption style="text-align: center;"><em>Phân bố số từ vựng sai chính tả</em></figcaption>
</figure>

##### Nhận xét:
  - Số lượng bài luận mắc lỗi từ 0 - 20 từ là nhiều nhất và tập trung ở mức điểm 2 - 4.
  - Hầu như các bài luận được điểm càng cao thì càng mắc ít lỗi chính tả.
  - Các điểm ngoại lai (các bài mắc rất nhiều lỗi - khoảng trên 60 lỗi) có điểm từ 1 - 4.
  - Bài mắc nhiều lỗi nhất (trên 100 lỗi) thường có điểm là 1.
#### WordCloud
<figure>
      <img title="a title" alt="Alt text" src="rpimg/wordcloud.svg">
      <figcaption style="text-align: center;"><em>WordCloud</em></figcaption>
 </figure>

##### Nhận xét
- Có thể thấy chủ đề của các bài luận xoay quanh driverless car, Electoral College, Seagoing Cowboy, Coding System, Face Mar,...
- Ta xem xét bigram của các bài luận có score là 6 và 1 để so sánh.
<figure>
      <img title="a title" alt="Alt text" src="rpimg/top_bigrams.svg">
      <figcaption style="text-align: center;"><em>Top BiGrams</em></figcaption>
 </figure>

##### Nhận xét:
- Những bigrams được sử dụng ở cả hai mức điểm là electoral college, popular vote và là những từ chủ đề như đã phân tích Wordcloud.
- Hầu như các bigrams nằm trong mức điểm 6 lại rất hiếm khi xuất hiện trong mức điểm 1, có thể vì thế nên các bài luận có score 1 không có tính thống nhất với chủ đề, do đó có số điểm thấp hơn.
#### Phân tích cảm xúc (Sentiment Analysis)
<figure>
      <img title="a title" alt="Alt text" src="rpimg/sccater_sentiment.png">
      <figcaption style="text-align: center;"><em>Sentiment Polarity</em></figcaption>
 </figure>

##### Nhận xét:
- Nếu xét các mức sentiment <0: Thì nhận thấy các bài essay điểm càng cao thì có sentiment càng cao.
- Ngược lại ở các mức sentiment >0: Hầu như các bài essay điểm càng thấp thì có sentiment càng cao.
=> Càng bài essay điểm càng cao có miền sentiment càng thấp.

#### Phân tích độ đa dạng từ vựng
<figure>
      <img title="a title" alt="Alt text" src="rpimg/simple_ttr.svg">
      <figcaption style="text-align: center;"><em>Histogram chỉ số simple TTR </em></figcaption>
</figure>

##### Nhận xét:
- Đa phần các bài essay có chỉ số đa dạng từ vựng (Simple TTR) phân bố nhiều ở khoảng (0.4-0.5) .Ta hãy cùng xem mối tương quan giữa chúng với điểm số của các bài essay.
<figure>
      <img title="a title" alt="Alt text" src="rpimg/ttr_score.svg">
      <figcaption style="text-align: center;"><em>Mối quan hệ chỉ số simple ttr và score</em></figcaption>
</figure>

##### Nhận xét:
- Không có mối quan hệ tỉ lệ thuận giữa điểm số và độ đa dạng từ vựng .Trong trường hợp datasets này có thể có cách chấm điểm không dựa vào độ đa dạng từ hoặc có thể các bài có nhiều từ nhưng tác giả lại sử dụng sai ngữ cảnh dẫn đến điểm có thể sẽ không cao.
- Tuy nhiên để đạt điểm cao (từ điểm 5 trở lên) thì hầu như các bài essay phải có chỉ số `Simple TTR` từ điểm 0.25 trở lên.

## 2. Transformer models dùng cho xử lý lý ngôn ngữ tự nhiên (https://huggingface.co/learn/nlp-course/chapter1/1)
### 2.1. Định nghĩa, các thức hoạt động
### 2.2. Phân loại (3 loại)
### 2.3. Cấu trúc pipelines (chapter  2)	
## 3. Fine tuning mô hình DeBERTaV3
### 3.1. Giới thiệu mô hình 
- Mục tiêu của bài toán là từ bài luận của học sinh, dự đoán điểm số từ 1 đến 6. Đây chính là một bài toán encoding điển hình. Các mô hình điển hình trong họ encoding như ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa, ... Trong đó có một mô hình khá nổi trội được sử dụng bởi khá nhiều nhóm tham gia cuộc thi là DeBERTaV3.

- DeBERTa cải thiện các mô hình BERT và RoBERTa bằng cách sử dụng cơ chế tập trung disentangled và bộ giải mã mặt nạ cải tiến. Trong DeBERTa V3, các tác giả đã cải thiện hiệu quả của DeBERTa bằng cách sử dụng tiền huấn luyện kiểu ELECTRA với Chia sẻ Nhúng Gradient-Disentangled. So với DeBERTa, phiên bản V3 của các tác giả cải thiện đáng kể hiệu suất mô hình trên các tác vụ downstream. 

- Một số task quan trọng mà mô trình encoding của transformer cung cấp bao gồm: Fill-Mask, Zere-shot Classification, Named Entity Recognition, Sentiment Analyzing, ... Bản thân DeBERTaV3 được pre-trained để phục vụ tác vụ Fill-Mask nhưng thư viện Transformers đã cung cấp đầy đủ các công cụ để ta có thể fine-tuning mô hình cho các tác vụ classification.

- Cụ thể ta sẽ dùng lênh sau:
```python
AutoModelForSequenceClassification.from_pretrained(cfg.checkpoint,num_labels=1)
```
- Trong đó:
    - `cfg.checkpoint`: checkpoint của mô hình DeBERTaV3 (có thể là local hoặc trên huggingface).
    - `num_labels=1`: Nhóm dự định sẽ để mô hình dự đoán số thực từ 1 đến 6 nên ta sẽ chỉ cần một label.
- Sau khi chạy lệnh này ta sẽ nhận được thông báo 
```
Some weights of DebertaV2ForSequenceClassification were not initialized from the model checkpoint at /kaggle/input/init-aes2/microsoft__deberta-v3-small and are newly initialized: ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight'] 
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```
- Có nghĩa là ta sẽ không thể tận dụng được các trọng số đã được pre-trained của mô hình nên ta cần phải fine-tuning một mô hình hoàn toàn mới trên tập dữ liệu của mình.

- Mô hình DeBERTaV3 sẽ có 4 phiên bản với độ lớn giảm dần như sau:
    - DeBERTaV3-large: mô hình quá lớn gây ra tràn VRAM nên không thể sử dụng được.
    - DeBERTaV3-base: Mô hình cho kết quả tốt nhất, được nhóm sử dụng để làm bài nộp. Mỗi epoch của mô hình này mất khoảng 30 phút chạy
    - DeBERTaV3-small: Mô hình cho tốc độ chạy nhanh hơn (20 phút/epoch) và kết quả cũng khá tốt nên được nhóm sử dụng chủ yếu để thử nghiệm.
    - DeBERTaV3-xsmall: Mô hình nhỏ nhất, được nhóm sử dụng để khởi đầu.
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
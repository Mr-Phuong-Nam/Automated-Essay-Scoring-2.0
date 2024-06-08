# Transformer
Để một mô hình học máy có thể hiểu được ngôn ngữ của con người, từ đó xử lí các thông tin trên dữ liệu văn bản để đưa ra được dự đoán đúng nhất, thì **Transformer** chính là công cụ lí tưởng và tốt nhất cho hầu hết các tác vụ xử lí ngôn ngữ tự nhiên hiện nay, đây cũng chính là công cụ mà OpenAI đã sử dụng và tạo ra ChatGPT - một AI tạo sinh ngôn ngữ được hàng triệu người trên thế giới sử dụng. Sau đây chúng ta sẽ tìm hiểu sơ lược về khái niệm của mô hình Transformer như sau:

### Định nghĩa
Transformer là một kiến trúc mạng nơ-ron sâu, được giới thiệu lần đầu tiên trong bài báo "Attention is All You Need" của Vaswani và các cộng sự vào năm 2017. Nó đã trở thành một phương pháp chủ chốt trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP) và dịch máy tự động, cũng như nhiều ứng dụng khác trong học sâu.

### Kiến trúc cơ bản của Transformer
Transformer bao gồm hai thành phần chính: **Bộ mã hóa (Encoder)** và **Bộ giải mã (Decoder)**:

**Bộ mã hóa (Encoder):** Encoder nhận dữ liệu đầu vào và xây dựng các đặc trưng (features) của nó. Có nghĩa là mô hình được tối ưu hóa để hiểu biết các thông tin từ đầu vào

**Bộ giải mã (Decoder):** Decoder sử dụng các đặc trưng từ bộ mã hóa cùng với các đầu vào khác để tạo ra một chuỗi đích (target sequence). Điều này có nghĩa là mô hình được tối ưu hóa cho việc tạo ra kết quả.

<figure>
    <p align="center">
    <img title="a title" alt="Alt text" src="/rpimg/transformers.svg">
    </p>
    <figcaption style="text-align: center;"><em>Kiến trúc transformer</em></figcaption>
</figure>

Mỗi phần của mô hình có thể được sử dụng độc lập, tùy thuộc vào nhiệm vụ:
- **Mô hình chỉ mã hóa (Encoder-only models):** Phù hợp cho các nhiệm vụ cần hiểu biết về đầu vào, như phân loại câu và nhận dạng thực thể (Named Entity Recognition - NER).
- **Mô hình chỉ giải mã (Decoder-only models):** Phù hợp cho các tác vụ tạo sinh dữ liệu, như tạo văn bản.
- **Mô hình mã hóa-giải mã (Encoder-decoder models) hoặc mô hình chuỗi-sang-chuỗi (sequence-to-sequence models):** Phù hợp cho các nhiệm vụ tạo sinh dữ liệu nhưng cần nhận dữ liệu đầu vào, như các nhiệm vụ liên quan đến dịch thuật hoặc tóm tắt.

### Quy trình hoạt động tổng thể
1. **Mã hóa (Endcoding):**
- Chuỗi đầu vào được đưa qua nhiều lớp mã hóa
- Mỗi lớp mã hóa gồm self-attention và feed-forward network
- Kết quả cuối cùng là chuỗi biểu diễn của đầu vào đã được mã hóa
2. **Giải mã (Decoding):**
- Chuỗi đầu ra từng phần được đưa qua nhiều lớp giải mã
- Mỗi lớp giải mã gồm masked self-attention, encoder-decoder attention và feed-forward network.
- Kết quả cuối cùng là chuỗi đầu ra đã được giải mã hoàn chỉnh.

Bạn có thể tìm hiểu thêm về Transformer tại bài báo "Attention Is All You Need!" <em><a href="https://arxiv.org/pdf/1706.03762" style="text-decoration: none; color: inherit;">[https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)</a></em>

### Cấu trúc pipeline
Công cụ Pipeline: `pipeline()` là công cụ cơ bản nhất trong thư viện Transformers, kết nối mô hình với các bước tiền xử lý và hậu xử lý cần thiết, cho phép nhập trực tiếp văn bản và nhận câu trả lời có ý nghĩa.
Ta có ví dụ sau:
```
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

Sau khi thực thi đoạn mã trên ta có kết quả sau:
```
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```
Quy trình này kết hợp ba bước: *tiền xử lý (preprocessing), truyền dữ liệu qua mô hình (model),* và *hậu xử lý (post processing)*.

<figure>
    <img title="a title" alt="Alt text" src="/rpimg/pipeline.svg">
    <figcaption style="text-align: center;"><em>Tiến trình pipeline</em></figcaption>
</figure>

Ta sẽ đi qua từng quá trình trên

**Tiền xử lý (Preprocessing):** Bước này bao gồm việc làm sạch và chuẩn hóa dữ liệu, như loại bỏ nhiễu, chuẩn hóa chính tả, và phân tách từ (tokenization). Mục tiêu là chuẩn bị dữ liệu đầu vào để dễ dàng xử lý hơn.
```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```
*Đoạn code trên thực hiện quá trình xử lí token hóa cho tập dữ liệu đầu vào, với tokenizer từ một mô hình đã được huấn luyện sẵn (pretrained model).*

Kết quả đầu ra lúc này sẽ là:
```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

Kết quả là một từ điển chứa hai khóa, `input_ids` và `attention_mask`. `input_ids` chứa hai dòng số nguyên (một cho mỗi câu) là các định danh duy nhất của các từ trong mỗi câu, `attention_mask` chỉ ra những token nào được mô hình chú ý đến khi xử lý (giá trị 1 có nghĩa là token đó sẽ được tính toán, còn giá trị 0 có nghĩa là token đó sẽ bị bỏ qua, thường là do nó là padding (đệm) không có ý nghĩa)

**Mô hình hóa (Modeling):** Trong bước này, dữ liệu đã qua tiền xử lý được đưa qua một hoặc nhiều mô hình học máy để rút trích thông tin hoặc tạo ra ngôn ngữ. Các mô hình có thể bao gồm mô hình chỉ mã hóa, chỉ giải mã, hoặc cả mã hóa và giải mã.

```
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```
*Cách lấy một mô hình đã được huấn luyện để sử dụng cho dữ liệu đầu vào*

**Hậu xử lý (Postprocessing):** Sau khi mô hình đã xử lý dữ liệu, bước hậu xử lý sẽ chuyển kết quả của mô hình thành dạng cuối cùng mà người dùng có thể hiểu được. Điều này có thể bao gồm việc chuyển đổi kết quả thành văn bản tự nhiên hoặc thực hiện các điều chỉnh cuối cùng trên kết quả.
Sau khi đưa dữ liệu vào mô hình ở trên, ta có dự đoán sau:
```
print(outputs.logits)
```
```
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```
*Đây là các giá trị *logits* mà mô hình đã dự đoán từ dữ liệu đầu vào, với mỗi vector tương ứng với một chuỗi từ dữ liệu input.*

Vì *logits* chỉ là kết quả thô, chưa được chuẩn hóa, do đó ta phải sử dụng các kĩ thuật khác để đưa nó trở thành kết quả cuối cùng. Trong đoạn code trên, ta sẽ đưa *logits* qua một lớp hàm *Softmax* để phân tích quan điểm (sentiment analysis): 
```
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```
```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```
*Bây giờ chúng ta có thể thấy rằng mô hình đã dự đoán [0.0402, 0.9598] cho câu đầu tiên và [0.9995, 0.0005] cho câu thứ hai. Đó là điểm số xác suất mong muốn.*
Để có được các nhãn tương ứng với mỗi vị trí, chúng ta có thể kiểm tra thuộc tính `id2label` của cấu hình model:
```
model.config.id2label
```
```
{0: 'NEGATIVE', 1: 'POSITIVE'}
```
Bây giờ chúng ta có thể kết luận rằng mô hình dự đoán như sau:

Câu đầu tiên: `NEGATIVE`: 0.0402, `POSITIVE`: 0.9598
Câu thứ hai: `NEGATIVE`: 0.9995, `POSITIVE`: 0.0005
**Ứng dụng:** Transformer pipeline đã được áp dụng trong một loạt các nhiệm vụ NLP:
- **Phân loại văn bản:** Phân tích cảm xúc, phân loại chủ đề và phát hiện thư rác được xử lý một cách hiệu quả bởi đường ống phân loại văn bản.
- **Nhận dạng thực thể (NER):** Việc xác định các thực thể như tên, địa điểm, và ngày tháng trong văn bản được đơn giản hóa với đường ống NER.
- **Tạo văn bản:** Việc tạo văn bản mạch lạc và phù hợp với ngữ cảnh có thể truy cập được thông qua quy trình tạo văn bản.
- **Dịch thuật:** Việc dịch văn bản giữa các ngôn ngữ được thực hiện liền mạch bằng quy trình dịch thuật.
- **Trả lời câu hỏi:** Việc trích xuất câu trả lời từ một văn bản nhất định cho một câu hỏi nhất định được thực hiện bằng quy trình trả lời câu hỏi.
## Tài liệu tham khảo:
https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt

https://medium.com/@rakeshrajpurohit/exploring-hugging-face-transformer-pipelines-bd432220af0a

https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html

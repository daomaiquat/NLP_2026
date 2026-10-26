# Phân loại cảm xúc IMDB: RNN, LSTM, GRU và baseline học máy

So sánh các kiến trúc học sâu xử lý chuỗi (RNN, LSTM, GRU) với các baseline học máy truyền thống (SVM, MLP) trên bài toán phân loại cảm xúc review phim IMDB.

Điểm quan trọng: **mọi mô hình dùng chung một biểu diễn từ FastText đã đóng băng (frozen)**. Nhờ vậy, khác biệt về kết quả chủ yếu đến từ kiến trúc mô hình, không phải từ việc mô hình nào được học lại embedding tốt hơn.

## Tổng quan

- Dữ liệu: 50.000 review IMDB (25.000 positive, 25.000 negative).
- Chia tập: 40.000 review đầu cho train, 10.000 review cuối cho test.
- Từ 40.000 review train, tách tiếp 10% làm validation: 36.000 train / 4.000 validation.
- Vocabulary: 25.000 từ phổ biến nhất + `<PAD>` + `<UNK>`.
- Độ dài chuỗi: tối đa 150 token, pad/truncate về cùng độ dài.
- Embedding: FastText `fasttext-wiki-news-subwords-300`, vector 300 chiều.
- Ngưỡng phân loại: 0.5. Metric: Accuracy, Precision, Recall, F1-score.

## Biểu diễn từ và cách đóng băng embedding

### 1. Tạo ma trận embedding

Mỗi từ trong vocabulary được ánh xạ sang vector 300 chiều lấy từ FastText:

- `<PAD>` (index 0): vector 0.
- `<UNK>` (index 1) và các từ không khớp FastText: khởi tạo ngẫu nhiên.
- Ma trận `weights_matrix` có kích thước `25002 x 300`.

### 2. Đóng băng embedding

Tất cả mô hình neural đều tạo embedding từ `weights_matrix` với `freeze=True`:

```python
self.embedding = nn.Embedding.from_pretrained(
    embedding_weights, freeze=True, padding_idx=0
)
```

Khi đó:

- `embedding.weight.requires_grad = False`.
- Không có gradient chảy vào embedding.
- Trọng số embedding không bị cập nhật trong suốt quá trình train.
- Hàng `<PAD>` giữ nguyên vector 0.

Trong notebook, các model được khởi tạo với `freeze_embed=True`:

```python
model = First_GRU(
    embedding_weights=weights_matrix,
    hidden_dim=HIDDEN_DIM,
    freeze_embed=True,
)
```

Việc đóng băng còn tránh một lỗi tinh vi: `nn.Embedding.from_pretrained` có thể dùng chung vùng nhớ với `weights_matrix` (đặc biệt trên CPU), nên nếu để embedding trainable thì quá trình train có thể sửa trực tiếp `weights_matrix`. Khi đóng băng, mọi mô hình đều bắt đầu từ đúng một bộ FastText gốc giống nhau.

### 3. Baseline SVM và MLP dùng cùng embedding

Với SVM và MLP, mỗi review được biểu diễn bằng **mean pooling** trên các token không phải padding:

```python
keep = ids != 0
feature = weights_matrix[ids[keep]].mean(axis=0)
```

Kết quả là vector 300 chiều, nằm đúng trong không gian embedding mà RNN/LSTM/GRU sử dụng. Feature được chuẩn hóa bằng `StandardScaler` fit trên train.

Cách này giữ nguyên thông tin từ vựng nhưng làm mất thứ tự từ. Vì vậy, so sánh giữa MLP/SVM với RNN/LSTM/GRU cho thấy giá trị của việc mô hình hóa thứ tự chuỗi.

### 4. Các yếu tố được giữ cố định để so sánh công bằng

| Yếu tố | Cách kiểm soát |
|---|---|
| Dữ liệu và split | Cùng `train_subset`, `val_subset`, `test_dataset`, seed 42 |
| Tiền xử lý | Cùng `clean_text`, cùng vocabulary, cùng `MAX_LEN = 150` |
| Biểu diễn từ | Cùng `weights_matrix` FastText 300 chiều |
| Embedding | Đóng băng cho tất cả mô hình neural |
| Baseline học máy | Dùng mean-pool trên cùng embedding, bỏ qua PAD |
| Chuẩn hóa | `StandardScaler` fit trên train, áp dụng cho val/test |
| Ngưỡng | 0.5 cho mọi mô hình |
| Metric | Cùng Accuracy, Precision, Recall, F1-score |
| Chọn mô hình | Neural dùng early stopping trên validation F1; SVM tune `C` trên validation; MLP early stopping trên validation |

## Kiến trúc mô hình

| Mô hình | Mô tả |
|---|---|
| `First_RNN` | RNN thuần 1 tầng tự cài đặt, hidden 64, lấy hidden state cuối |
| `Second_RNN` | `nn.RNN` 2 tầng, 2 chiều, hidden 128, max + mean pooling, dropout 0.3, LayerNorm |
| `First_LSTM` | LSTM 1 tầng, hidden 64, lấy hidden state cuối |
| `Second_LSTM` | `nn.LSTM` 2 tầng, 2 chiều, hidden 128, max + mean pooling, dropout 0.3, LayerNorm |
| `First_GRU` | GRU tự cài đặt công thức cổng, hidden 64, lấy hidden state cuối |
| `Second_GRU` | `nn.GRU` 2 tầng, 2 chiều, hidden 128, max + mean pooling, dropout 0.3, LayerNorm |
| `SVM` | `LinearSVC` trên vector mean-pool 300 chiều, tune `C` trong `{0.01, 0.1, 1.0}` |
| `MLP` | `MLPClassifier` 1 hidden layer 256, early stopping thủ công trên validation F1 |

## Cấu hình huấn luyện

- Số epoch tối đa: 15.
- Batch size: 64.
- Optimizer:
  - `First_*`: Adam, learning rate `1e-3`, weight decay 0.
  - `Second_*`: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Gradient clipping: `max_norm = 1.0`.
- Scheduler: `ReduceLROnPlateau` theo validation F1, factor 0.5, patience 1.
- Early stopping: patience 3, theo validation F1, khôi phục trọng số tốt nhất.
- Hàm lỗi: `BCEWithLogitsLoss`.
- Test chỉ được đánh giá một lần sau khi chọn mô hình bằng validation.

## Kết quả

| Mô hình | Loss | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| First_RNN | 0.592931 | 0.7082 | 0.672845 | 0.812063 | 0.735928 |
| Second_RNN | 0.333578 | 0.8530 | 0.815860 | 0.912323 | 0.861399 |
| First_LSTM | 0.350022 | 0.8497 | 0.818777 | 0.898742 | 0.856898 |
| Second_LSTM | 0.308122 | 0.8731 | 0.862913 | 0.887557 | 0.875062 |
| First_GRU | 0.350608 | 0.8446 | 0.804444 | 0.911124 | 0.854467 |
| **Second_GRU** | **0.295137** | **0.8759** | **0.870232** | 0.883962 | **0.877043** |
| SVM (mean pool) | NaN | 0.7827 | 0.784996 | 0.779509 | 0.782243 |
| MLP (mean pool) | NaN | 0.8007 | 0.793533 | 0.813661 | 0.803471 |

`NaN` ở SVM và MLP vì hai mô hình này không tối ưu trực tiếp `BCEWithLogitsLoss` nên không có loss tương đương để so sánh.

## Nhận xét

- **Second_GRU đạt F1 cao nhất (0.8770)** và Accuracy cao nhất (0.8759), nhỉnh hơn Second_LSTM (F1 0.8751) nhưng chênh lệch không lớn.
- Nhóm `Second_*` (2 tầng, 2 chiều, max + mean pooling) vượt rõ nhóm `First_*` (1 tầng, lấy hidden cuối). Nguyên nhân chính là bidirectional, nhiều tầng và pooling trên toàn bộ timestep giúp giảm ảnh hưởng của vanishing gradient và tận dụng tốt hơn ngữ cảnh.
- GRU và LSTM cho kết quả gần nhau trên cùng cấu hình; GRU có ít tham số hơn nhưng không thua kém.
- Khi dùng **cùng embedding đã đóng băng**, các mô hình học sâu xử lý chuỗi vẫn vượt MLP/SVM. Điều này cho thấy phần cải thiện đến từ khả năng mô hình hóa thứ tự từ, không phải từ embedding được fine-tune.
- MLP (F1 0.8035) tốt hơn SVM (F1 0.7822) trên cùng vector mean-pool.
- `First_RNN` có F1 thấp nhất (0.7359), thấp hơn cả SVM, phù hợp với việc RNN thuần 1 tầng khó học phụ thuộc dài.
- Các mô hình `First_*` có recall cao nhưng precision thấp hơn, tức là thiên về dự đoán positive; `Second_*` cân bằng precision/recall hơn.

## Cách chạy

1. Cài thư viện:

```bash
pip install numpy pandas torch scikit-learn gensim
```

2. Đặt file `IMDB Dataset.csv` cùng thư mục với notebook, hoặc sửa `DATA_PATH` trong notebook.

3. Mở `rnn.ipynb` và chạy lần lượt các cell. Lần đầu chạy sẽ tải FastText `fasttext-wiki-news-subwords-300` (khoảng 1 GB).

## Cấu trúc repository

```text
NLP_2026/
└── rnn.ipynb
```

`IMDB Dataset.csv` không được đưa lên repository do kích thước lớn.

## Hạn chế

- Kết quả chỉ từ một lần chạy với seed 42, chưa báo cáo trung bình và độ lệch chuẩn qua nhiều seed.
- Vocabulary được xây trên toàn bộ 50.000 review trước khi chia train/test, nên có rò rỉ nhẹ. Tuy nhiên cách làm này giống nhau cho mọi mô hình nên không làm thay đổi bản chất so sánh tương đối.
- Tập test là 10.000 review cuối thay vì chia ngẫu nhiên.
- Review bị cắt còn 150 token đầu, có thể mất thông tin ở các review dài.
- SVM và MLP dùng mean pooling nên không khai thác được thứ tự từ; đây là baseline có chủ đích để làm nổi bật vai trò của mô hình chuỗi.

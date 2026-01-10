# Dự Án Phân Loại Cảm Xúc Bình Luận Tiếng Việt

## I. Giới Thiệu Tổng Quan

Đây là một dự án nghiên cứu về phân loại cảm xúc bình luận tiếng Việt, sử dụng các phương pháp Machine Learning truyền thống kết hợp với kỹ thuật giảm chiều dữ liệu. Dự án tập trung vào việc xây dựng và so sánh hiệu suất của nhiều mô hình khác nhau trên bài toán phân loại đa lớp với 7 nhãn cảm xúc.

**Mục tiêu chính:**
- Xây dựng bộ dữ liệu cảm xúc tiếng Việt từ 2 nguồn chính: UIT-VSMEC và Facebook
- Áp dụng pipeline tiền xử lý bình luận tiếng Việt
- Thử nghiệm và đánh giá 2 mô hình Machine Learning với 2 kỹ thuật giảm chiều khác nhau (PCA, LDA)
- So sánh hiệu quả của các phương pháp trên các tỷ lệ chia dữ liệu khác nhau

**Các mô hình được sử dụng:**
- Logistic Regression (Multinomial Softmax)
- K-Nearest Neighbors (K-NN) với Cosine Similarity

**Kỹ thuật giảm chiều:**
- PCA (Principal Component Analysis) - giảm chiều không giám sát
- LDA (Linear Discriminant Analysis) - giảm chiều có giám sát

---

## II. Nguồn Dữ Liệu

### 1. UIT-VSMEC Dataset

**Mô tả:** 
UIT-VSMEC (Vietnamese Social Media Emotion Corpus) là bộ dữ liệu chuẩn về phân loại cảm xúc tiếng Việt được phát triển bởi Trường Đại học Bách khoa - ĐHQG TP.HCM. Bộ dữ liệu này tập trung vào các bình luận trên mạng xã hội với ngôn ngữ tự nhiên, phi chính thức (từ lóng, từ viết tắt).

**Nguồn:** Hugging Face Datasets (`ura-hcmut/UIT-VSMEC`)

**Thông tin chi tiết:**
- Tập huấn luyện (Train): 5548 mẫu
- Tập kiểm định (Validation): 686 mẫu
- Tập kiểm tra (Test): 693 mẫu
- Tổng cộng: **6927 mẫu**

### 2. Bộ Dữ Liệu Tự Thu Thập và Gán Nhãn (HUSFBcGp)

**Mô tả:**
Ngoài UIT-VSMEC, dự án còn bổ sung bộ dữ liệu được tự thu thập từ mạng xã hội Facebook và được gán nhãn thủ công bởi nhóm. Việc thu thập và gán nhãn tuân theo cùng tiêu chuẩn với UIT-VSMEC để đảm bảo tính nhất quán.

**Quy trình tóm tắt:**
- Thu thập bình luận tiếng Việt từ mạng xã hội Facebook
- Làm sạch và lọc các mẫu không hợp lệ
- Gán nhãn thủ công bởi nhiều người để đảm bảo độ chính xác
- Kiểm tra chéo và đồng thuận về nhãn cuối cùng

**Thông tin chi tiết:**
- Số lượng mẫu: **4616 mẫu**
- File dữ liệu: `data/raw_data_final.csv`

### 3. Bộ Dữ Liệu Tổng Hợp (UIT-VSMEC-HUSFBcGp)

**Sau khi gộp hai nguồn dữ liệu:**
- Tổng số mẫu *(chưa tiền xử lý dữ liệu)*: **11543 mẫu**
- Sau khi tiền xử lý, EDA và loại bỏ ngoại lai: **10780 mẫu**
- Số nhãn cảm xúc: **7 nhãn**
- Phân bố dữ liệu: Không cân bằng (imbalanced), nhãn "Enjoyment" chiếm đa số

**Các nhãn cảm xúc:**
1. **Enjoyment** - Thích thú, vui vẻ
2. **Sadness** - Buồn bã
3. **Anger** - Tức giận
4. **Fear** - Sợ hãi
5. **Surprise** - Ngạc nhiên
6. **Disgust** - Ghê tởm
7. **Other** - Cảm xúc khác

**Chia tập dữ liệu:**
- Train + Validation: 8624 mẫu (80%)
- Test: 2156 mẫu (20%)
- Các tỷ lệ Train:Validation được thử nghiệm: 8:2, 7:3, 6:4

---

## III. Quy Trình Tiền Xử Lý Dữ Liệu

Dự án áp dụng một pipeline tiền xử lý 13 bước cho bình luận tiếng Việt:

### 1. Làm Sạch Cơ Bản (Bước 1-5)

1. **Loại bỏ null khỏi dữ liệu**
   - Xóa các dòng dữ liệu không có nội dung

2. **Chuẩn hóa chữ thường**
   - Chuyển toàn bộ bình luận về chữ thường để đồng nhất

3. **Chuyển ký tự lặp lại liên tiếp thành 1 ký tự đơn**
   - "đẹpppp" => "đẹp"
   - "yayyyy" => "yay"

4. **Thay thế các Emoji và Emoticon thành từ tiếng Việt**
   - `:)` => `<cười>`
   - `😂` => `<cười_chảy_nước_mắt>`
   - `❤️` => `<tim>`
   - `T_T` => `<khóc>`

5. **Chuẩn hóa từ viết tắt và tiếng lóng**
   - "ko" => "không"
   - "k" => "không"
   - "oke" => "ok"

### 2. Bước 6-9

6. **Xóa các dòng có dấu thanh bị tách rời**
   - Loại bỏ các bình luận bị lỗi Unicode (dấu thanh tách rời khỏi chữ cái)

7. **Xóa ký tự đặc biệt**
   - Giữ lại các ký tự chữ cái, số, và một số dấu câu cần thiết (`_`, `.`, `;`, `,`, `!`, `?`)

8. **Chuẩn hóa chính tả với Underthesea**
   - Sử dụng thư viện `underthesea` để chuẩn hóa chính tả tiếng Việt

9. **Loại bỏ trùng lặp, xét trên cột Sentence**
   - Xóa các mẫu có nội dung giống hệt nhau

### 3. Tokenization và Hoàn Thiện (Bước 10-13)

10. **Tách từ (Word Tokenization)**
    - Sử dụng `underthesea` để tách từ tiếng Việt
    - "Hôm nay trời đẹp quá" => ["Hôm_nay", "trời", "đẹp", "quá"]

11. **Loại bỏ dấu câu sau tokenization**
    - Loại bỏ các dấu câu không cần thiết sau khi đã tách từ

12. **Loại bỏ từ dừng (Stopwords)**
    - Loại bỏ các từ phổ biến không mang nhiều ý nghĩa: "và", "là", "có", "của", "cho", "với", v.v.

13. **Loại bỏ dòng rỗng**
    - Loại bỏ các dòng trở thành rỗng sau quá trình xử lý

**Thư viện sử dụng:**
- `underthesea` - Xử lý ngôn ngữ tiếng Việt
- `pandas` - Xử lý dữ liệu dạng bảng
- `re` (regex) - Xử lý chuỗi với biểu thức chính quy

---

## IV. Cấu Trúc Thư Mục

```
0_final/
│
├── README.md                          # Tài liệu hướng dẫn dự án
│
├── data/                              # Thư mục chứa dữ liệu
│   ├── raw_data_final.csv             # Dữ liệu thô sau khi gộp 2 bộ dữ liệu
│   ├── final_checked_data.csv         # Dữ liệu tự gán nhãn (HUSFBcGp)
│   │
│   ├── cleaned_final/                 # Dữ liệu sau tiền xử lý
│   │   ├── processed_comments.csv     # Toàn bộ dữ liệu đã xử lý (chưa loại bỏ ngoại lai)
│   │   ├── train_valid_processed_comments.csv  # Tập train+valid (80%)
│   │   └── test_processed_comments.csv         # Tập test (20%)
│   │
│   └── UIT_VSMEC/                     # Dataset gốc từ UIT
│       ├── vsmec_merged.csv           # Gộp train+valid+test
│       ├── vsmec_train.csv            # Tập train gốc
│       ├── vsmec_valid.csv            # Tập validation gốc
│       └── vsmec_test.csv             # Tập test gốc
│
├── data_build.ipynb                   # Notebook 1: Xây dựng bộ dữ liệu
├── preprocessing.ipynb    # Notebook 2: Tiền xử lý dữ liệu
├── eda.ipynb                          # Notebook 3: Phân tích khám phá dữ liệu
├── pca_lda.ipynb                      # Notebook 4: Thử nghiệm giảm chiều
└── modeling.ipynb               # Notebook 5: Huấn luyện và đánh giá mô hình
```

---

## V. Pipeline Thực Nghiệm

### 1. Tổng Quan Pipeline

Dự án thực hiện một pipeline hoàn chỉnh từ xây dựng dữ liệu đến đánh giá mô hình:

```
[Xây dựng dữ liệu] => [Tiền xử lý] => [EDA] => [TF-IDF, PCA, LDA] => [Modeling] => [Evaluation]
```

### 2. Chi Tiết Các Bước

#### Bước 1: Xây Dựng Bộ Dữ Liệu (`data_build.ipynb`)

**Mục tiêu:** Xây dựng bộ dữ liệu cảm xúc tiếng Việt từ 2 nguồn chính: UIT-VSMEC và Facebook

**Quy trình:**
1. Tải UIT-VSMEC dataset từ Hugging Face
2. Đọc bộ dữ liệu tự gán nhãn (`final_checked_data.csv`)
3. Gộp hai nguồn dữ liệu thành một
4. Lưu kết quả vào `raw_data_final.csv`

**Output:**
- `data/raw_data_final.csv` - Dữ liệu thô tổng hợp
- `data/UIT_VSMEC/*.csv` - Các file dataset gốc

#### Bước 2: Tiền Xử Lý Dữ Liệu (`preprocessing_final_final.ipynb`)

**Mục tiêu:** Làm sạch và chuẩn hóa bình luận tiếng Việt

**Quy trình:** Áp dụng 13 bước tiền xử lý *(xem phần "Quy Trình Tiền Xử Lý Dữ Liệu")*

**Output:**
- `data/cleaned_final/processed_comments.csv` - Dữ liệu đã tiền xử lý hoàn chỉnh

#### Bước 3: Phân Tích Khám Phá Dữ Liệu (`eda.ipynb`)

**Mục tiêu:** Hiểu sâu về đặc điểm và phân bố dữ liệu

**Các phân tích thực hiện:**

1. **Tổng quan**
   - Số lượng mẫu sau khi tiền xử lý: **11,380**
   - Số nhãn: 7
   - Phân bố nhãn: Không cân bằng (Enjoyment chiếm đa số)

2. **Phân tích độ dài câu**
   - Tính độ dài token cho mỗi câu
   - Phát hiện outliers bằng phương pháp IQR
   - Loại bỏ 41 mẫu có độ dài > 500 tokens
   - Ngưỡng dưới: -46.875, ngưỡng trên: 174.125

```
Số lượng dòng loại bỏ sau khi loại bỏ outliner: 600
Kích thước bộ dữ liệu sau khi loại bỏ outliner: (10780, 4)
```   

3. **Phân tích từ và cột Emotion (nhãn) sau khi loại bỏ outliers**
   - Vẽ WordCloud cho từng nhãn cảm xúc
   - Thống kê số lượng nhãn

4. **Chia tập dữ liệu**
   - Train+Valid : Test = 8:2 (stratified split)
   - Đảm bảo phân bố nhãn tương đương giữa các tập

**Output:**
- `data/cleaned_final/train_valid_processed_comments.csv` - 8624 mẫu
- `data/cleaned_final/test_processed_comments.csv` - 2156 mẫu
- Các biểu đồ phân tích trực quan

#### Bước 4: Thử Nghiệm Giảm Chiều (`pca_lda.ipynb`)

**Mục tiêu:** Khám phá hiệu quả của các kỹ thuật giảm chiều

**Kỹ thuật được thử nghiệm:**

1. **PCA (Principal Component Analysis)**
   - Phương pháp giảm chiều không giám sát
   - Giữ lại 90% phương sai của dữ liệu

2. **LDA (Linear Discriminant Analysis)**
   - Phương pháp giảm chiều có giám sát
   - Tối đa hóa khoảng cách giữa các lớp
   - Kết quả: Giảm xuống 6 chiều (n_classes - 1 = 7 - 1 = 6)

#### Bước 5: Huấn Luyện và Đánh Giá Mô Hình (`modeling_final.ipynb`)

**Mục tiêu:** Xây dựng và so sánh các mô hình phân loại

**Pipeline Modeling Hoàn Chỉnh:**

```python
pipeline_modeling(
    df_train_valid,          # DataFrame train+validation
    df_test,                 # DataFrame test
    train_size,              # Tỷ lệ train (8, 7, hoặc 6)
    valid_size,              # Tỷ lệ valid (2, 3, hoặc 4)
    random_state=11,
    vectorizer="tf_idf", min_df=2, max_df=0.95, max_features=8000,
    discriminant="none",     # "none", "pca", hoặc "lda"
    model_type="logistic",   # "logistic_regression" hoặc "knn"
    n_components_pca=0.9,    # Giữ 90% variance cho PCA
    n_components_lda=6,      # 6 chiều cho LDA
    max_iter=1000,           # Số lần lặp tối đa
    metric='cosine'          # metric trong knn
)
```

**Các bước trong pipeline:**

1. **Chia dữ liệu**
   - Tách train-validation-test theo tỷ lệ chỉ định
   - Sử dụng stratified split để đảm bảo phân bố nhãn

2. **TF-IDF Vectorization**
   - `max_features=8000` - Chỉ giữ 8000 từ phổ biến nhất
   - `ngram_range=(1, 2)` - Sử dụng unigram và bigram
   - `min_df=2` - Từ phải xuất hiện ít nhất 2 lần
   - `max_df=0.95` - Loại bỏ từ xuất hiện quá 95% documents

3. **Giảm chiều (tùy chọn)**
   - Áp dụng PCA hoặc LDA nếu được chỉ định
   - Chuyển đổi dữ liệu sang không gian mới

4. **Huấn luyện mô hình**
   - Logistic Regression với class_weight='balanced'
   - K-NN với n_neighbors=83 và metric='cosine'

5. **Dự đoán và đánh giá**
   - Dự đoán trên cả 3 tập: train, validation, test
   - Tính toán các metrics

**Mô hình Machine Learning:**

1. **Logistic Regression (Multinomial Softmax)**
   ```python
   LogisticRegression(
       multi_class='multinomial',    # Softmax cho đa lớp
       class_weight='balanced',      # Xử lý imbalanced data
       max_iter=1000                 # Số lần lặp
   )
   ```
   - Bài toán phân loại đa lớp

2. **K-Nearest Neighbors (K-NN)**
   ```python
   KNeighborsClassifier(
       n_neighbors=83,     # Số lượng láng giềng (tối ưu từ grid search)
       metric='cosine',    # Sử dụng cosine similarity
       n_jobs=-1          # Sử dụng tất cả CPU cores
   )
   ```
   - Dựa trên nguyên lý "láng giềng gần nhất"
   - Cosine similarity

---

## VI. Kịch Bản Thực Nghiệm

Dự án thực hiện **18 kịch bản thực nghiệm** khác nhau, kết hợp:
- **3 tỷ lệ chia dữ liệu:** 8:2, 7:3, 6:4 (Train:Validation)
- **3 phương pháp xử lý chiều:** None, PCA, LDA
- **2 mô hình:** Logistic Regression, K-NN

### Ma Trận Thực Nghiệm

| ID | Train:Valid | Giảm Chiều | Mô Hình | Mô Tả |
|----|-------------|------------|---------|--------|
| 1  | 8:2 | None | Logistic | Baseline Logistic Regression với 8:2 |
| 2  | 8:2 | PCA | Logistic | Giảm chiều không giám sát |
| 3  | 8:2 | LDA | Logistic | Giảm chiều có giám sát |
| 4  | 8:2 | None | K-NN | Baseline với K-NN |
| 5  | 8:2 | PCA | K-NN | K-NN với PCA |
| 6  | 8:2 | LDA | K-NN | K-NN với LDA |
| 7  | 7:3 | None | Logistic | Tăng tập validation (7:3) |
| 8  | 7:3 | PCA | Logistic | 7:3 với PCA |
| 9  | 7:3 | LDA | Logistic | 7:3 với LDA |
| 10 | 7:3 | None | K-NN | 7:3 baseline K-NN |
| 11 | 7:3 | PCA | K-NN | 7:3 K-NN với PCA |
| 12 | 7:3 | LDA | K-NN | 7:3 K-NN với LDA |
| 13 | 6:4 | None | Logistic | Tăng tập validation (6:4) |
| 14 | 6:4 | PCA | Logistic | 6:4 với PCA |
| 15 | 6:4 | LDA | Logistic | 6:4 với LDA |
| 16 | 6:4 | None | K-NN | 6:4 baseline K-NN |
| 17 | 6:4 | PCA | K-NN | 6:4 K-NN với PCA |
| 18 | 6:4 | LDA | K-NN | 6:4 K-NN với LDA |

### Ví Dụ Chạy Thực Nghiệm

**Kịch bản 2: Logistic Regression với PCA, tỷ lệ 8:2**

```python
model, train, valid, test, pred = pipeline_modeling(
    train_valid_processed_comments,
    test_processed_comments,
    train_size=8,
    valid_size=2,
    discriminant="pca",
    n_components_pca=0.9,
    model_type="logistic_regression",
    max_iter=1000
)

# Đánh giá
pipeline_evaluation(
    train[1], valid[1], test[1],
    pred[0], pred[1], pred[2],
    train_size=8, valid_size=2,
    discriminant="pca",
    model_type="logistic_regression",
    show_plot=True
)
```

### Các Metrics Đánh Giá

Dự án sử dụng 4 metrics chính:

1. **Accuracy (Độ chính xác)**
   - Tỷ lệ dự đoán đúng trên tổng số mẫu
   - Công thức: (TP + TN) / (TP + TN + FP + FN)

2. **Precision (Độ chính xác dương)**
   - Tỷ lệ dự đoán đúng trong các mẫu được dự đoán là positive
   - Công thức: TP / (TP + FP)
   - Sử dụng weighted average cho đa lớp

3. **Recall (Độ bao phủ)**
   - Tỷ lệ dự đoán đúng trong các mẫu thực tế là positive
   - Công thức: TP / (TP + FN)
   - Sử dụng weighted average cho đa lớp

4. **F1-Score (Điểm F1)**
   - Trung bình điều hòa của Precision và Recall
   - Công thức: 2 * (Precision * Recall) / (Precision + Recall)
   - Cân bằng giữa precision và recall

**Trực quan hóa:**
- Confusion Matrix (Ma trận nhầm lẫn) được chuẩn hóa
- Hiển thị cho cả 3 tập: Train, Validation, Test

---

## VII. Hướng Dẫn Sử Dụng

### 1. Yêu Cầu Hệ Thống

**Môi trường:**
- Python 3.8 trở lên
- Jupyter Notebook hoặc JupyterLab
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)
- CPU: Multi-core (K-NN yêu cầu nhiều CPU)

**Thư viện cần thiết:**

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install underthesea
pip install datasets
pip install wordcloud
pip install gensim
```

### 2. Cách Chạy Dự Án

#### Chạy Toàn Bộ Pipeline (Từ Đầu)

**Bước 1: Xây dựng dữ liệu**
```bash
# Mở và chạy data_build.ipynb
jupyter notebook data_build.ipynb
```
- Tải UIT-VSMEC từ Hugging Face
- Gộp với dữ liệu tự gán nhãn
- Output: `data/raw_data_final.csv`

**Bước 2: Tiền xử lý dữ liệu**
```bash
# Mở và chạy preprocessing_final_final.ipynb
jupyter notebook preprocessing.ipynb
```
- Áp dụng 13 bước tiền xử lý
- Output: `data/cleaned_final/processed_comments.csv`

**Bước 3: Phân tích dữ liệu (EDA)**
```bash
# Mở và chạy eda.ipynb
jupyter notebook eda.ipynb
```
- Phân tích thống kê và trực quan hóa
- Loại bỏ outliers
- Chia tập train-validation-test
- Output: `train_valid_processed_comments.csv` và `test_processed_comments.csv`

**Bước 4: Huấn luyện mô hình**
```bash
# Mở và chạy modeling_final.ipynb
jupyter notebook modeling.ipynb
```
- Chạy 18 kịch bản thực nghiệm
- Đánh giá và so sánh kết quả

#### Chạy Nhanh (Chỉ Modeling)

Nếu dữ liệu đã được xử lý sẵn trong thư mục `data/cleaned_final/`:

```python
# Mở modeling_final.ipynb và chạy từ đầu
# Chọn kịch bản thực nghiệm mong muốn

# Ví dụ: Chạy kịch bản Logistic + PCA + 8:2
model, train, valid, test, pred = pipeline_modeling(
    train_valid_processed_comments,
    test_processed_comments,
    train_size=8,
    valid_size=2,
    discriminant="pca",
    model_type="logistic_regression",
    max_iter=1000
)
```

### Tùy Chỉnh Thực Nghiệm

**Thay đổi tỷ lệ chia dữ liệu:**
```python
# 8:2 - Nhiều dữ liệu training
train_size=8, valid_size=2

# 7:3 - Cân bằng
train_size=7, valid_size=3

# 6:4 - Nhiều dữ liệu validation
train_size=6, valid_size=4
```

**Chọn phương pháp giảm chiều:**
```python
# Không giảm chiều
discriminant="none"

# PCA giữ 90% variance
discriminant="pca", n_components_pca=0.9

# LDA giảm xuống 6 chiều
discriminant="lda", n_components_lda=6
```

**Chọn mô hình:**
```python
# Logistic Regression
model_type="logistic_regression"

# K-Nearest Neighbors
model_type="knn"
```

**Tùy chỉnh TF-IDF:**
```python
# Trong hàm pipeline_modeling, có thể thay đổi:
max_features=8000,  # Số từ vựng tối đa
min_df=2,           # Từ xuất hiện tối thiểu
max_df=0.95         # Từ xuất hiện tối đa
```

---

## VIII. Kết Quả và Phân Tích

### Biểu Đồ Confusion Matrix

Mỗi thực nghiệm tạo ra 3 confusion matrices:
1. **Train set** - Đánh giá overfitting
2. **Validation set** - Điều chỉnh hyperparameters
3. **Test set** - Đánh giá cuối cùng

### Hướng Cải Thiện

**Về dữ liệu:**
- Thu thập thêm dữ liệu cho các nhãn thiểu số
- Cải thiện chất lượng gán nhãn

**Về mô hình:**
- Thử nghiệm các mô hình deep learning (LSTM, Transformer)
- Sử dụng pre-trained models (PhoBERT, ViT5)

**Về đánh giá:**
- Áp dụng k-fold cross-validation
- Grid search cho hyperparameters
- Phân tích chi tiết các lỗi phân loại

---

## Ĩ. Tài Liệu Tham Khảo

**Dataset:**
- UIT-VSMEC: [https://huggingface.co/datasets/ura-hcmut/UIT-VSMEC](https://huggingface.co/datasets/ura-hcmut/UIT-VSMEC)

**Thư viện:**
- Underthesea: [https://github.com/undertheseanlp/underthesea](https://github.com/undertheseanlp/underthesea)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
- Hugging Face Datasets: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)

**Phương pháp:**
- TF-IDF: Term Frequency-Inverse Document Frequency
- PCA: Principal Component Analysis
- LDA: Linear Discriminant Analysis
- Logistic Regression: Multinomial Softmax
- K-NN: K-Nearest Neighbors with Cosine Similarity
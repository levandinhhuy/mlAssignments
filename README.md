# Bài Tập Lớn Học Máy – CO3117 (Nhóm 11, Lớp L01)
## Thông tin môn học

- Tên môn học: Học máy
- Mã môn học: CO3117
- Lớp: L01, nhóm 11
- Học kỳ: 252, năm học 2025 - 2026

## Giảng viên hướng dẫn

- TS. Trương Vĩnh Lân

## Thành viên nhóm

- Phạm Trần Đức Hạnh - 2310896
- Ngô Minh Huấn - 2311117
- Nguyễn Quốc Gia Huy - 2311215
- Lê Văn Đình Huy - 2311160 

## Mục tiêu bài tập lớn

1. Hiểu và áp dụng được quy trình pipeline học máy truyền thống, bao gồm: tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện và đánh giá mô hình.
2. Rèn luyện kỹ năng triển khai mô hình học máy trên các loại dữ liệu khác nhau: bảng, văn bản, và ảnh.
3. Phát triển khả năng phân tích, so sánh, và đánh giá hiệu quả của các mô hình học máy thông qua các chỉ số đo lường.
4. Rèn luyện kỹ năng lập trình, thử nghiệm, và tổ chức báo cáo khoa học.

## Dataset

Các tập dữ liệu được nhóm sử dụng trong Assignment này:

- **Bài 1 - Dữ liệu dạng bảng (Tabular Data):** [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists/data?select=aug_train.csv)

- **Bài 2 - Dữ liệu văn bản (Text Data):** [ArXiv Multi-label Text Classification](https://www.kaggle.com/datasets/kelixirr/arxiv-multi-label-text-classification-datasets?select=arxiv34k6L.csv)

- **Bài 3 - Dữ liệu hình ảnh (Image Data):** [Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification/data?select=train.csv)

## Notebook

- [Link notebook Assignment 1](https://colab.research.google.com/drive/1ZiUyHjCq-4apm0_fB8GvbpqxjNT14MwH?usp=sharing)
- [Link notebook Assignment 2](https://colab.research.google.com/drive/15qWFpY1L6CvhYGzelnHAHitxXtzSVf7c?usp=sharing)
- [Link notebook Assignment 3](https://colab.research.google.com/drive/1y-SDeWSbtbdgIdz9F8z2-zqZqSdzVgLz?usp=sharing)

## Hướng dẫn chạy notebook

- Mở notebook muốn chạy trong Google Colab.
- Chọn Runtime → Run All.
- Notebook đã được cấu hình sẵn: import thư viện, tải dataset, xử lý và chạy mô hình.
- Sau khi chạy, kết quả huấn luyện và đánh giá sẽ hiện ra.

## Cấu trúc thư mục

```text
mlAssignments/
├── README.md
├── Assignment1/
│   ├── data/
│   │   └── aug_train.csv
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── features_extractor.py
│   │   └── model_runner.py
│   └── notebooks/
│       └── Assignment1.ipynb
├── Assignment2/
│   ├── data/
│   │   └── arxiv34k6L.csv
│   ├── features/
│   │   ├── bow/
│   │   │   ├── Xte.npz
│   │   │   ├── Xtr.npz
│   │   │   └── Xva.npz
│   │   ├── tfidf/
│   │   │   ├── Xte.npz
│   │   │   ├── Xtr.npz
│   │   │   └── Xva.npz
│   │   └── tfidf_glove/
│   │       ├── Xte_w2v.npy
│   │       ├── Xtr_w2v.npy
│   │       └── Xva_w2v.npy
│   ├── modules/
│   │   ├── features_extractor.py
│   │   └── tfidf_glove.py
│   └── notebooks/
│       └── Assignment2.ipynb
└── Assignment3/
	├── data/
	│   ├── train.csv
	│   └── PaddyDisease/
	├── features/
	│   ├── efficientnetb0/
	│   │   ├── X_test.npy
	│   │   ├── X_train.npy
	│   │   ├── X_val.npy
	│   │   ├── y_test.npy
	│   │   ├── y_train.npy
	│   │   └── y_val.npy
	│   ├── resnet50/
	│   │   ├── X_test.npy
	│   │   ├── X_train.npy
	│   │   ├── X_val.npy
	│   │   ├── y_test.npy
	│   │   ├── y_train.npy
	│   │   └── y_val.npy
	│   └── vgg16/
	│       ├── X_test.npy
	│       ├── X_train.npy
	│       ├── X_val.npy
	│       ├── y_test.npy
	│       ├── y_train.npy
	│       └── y_val.npy
	├── modules/
	│   └── features_extractor.py
	└── notebooks/
		└── Assignment3.ipynb
```

## Hoạt động nhóm: [Các buổi họp](https://drive.google.com/drive/folders/123o4-E0HK8aXxGu-_SeLeesQq5W4o5ZE?usp=drive_link)

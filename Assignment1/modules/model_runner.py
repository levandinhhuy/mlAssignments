import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_model(model, num_feature, cat_feature, data, target, feature_range=(0, 1), test_size=0.2, random_state=42, n_neighbors=5, imputer='knn', scaler='minmax', n_components=0.95):

    """
    Hàm chạy pipeline: Tiền xử lý → PCA → Huấn luyện model → Đánh giá
    
    Parameters
    ----------
    model : sklearn estimator
        Mô hình học máy cần huấn luyện (LogisticRegression, SVC, RandomForestClassifier, ...)
    num_feature : list
        Danh sách các cột số (numeric features).
    cat_feature : list
        Danh sách các cột phân loại (categorical features).
    data : pandas.DataFrame
        Dữ liệu đã làm sạch.
    target : str
        Tên cột nhãn cần dự đoán.
    ...
    """

 # ------------------------------
 # 1. Chuẩn bị dữ liệu
 # ------------------------------
    x = data[num_feature + cat_feature]
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

# ------------------------------
# 2. Xử lý numeric và categorical
# ------------------------------
    # Imputer
    if imputer=='knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    elif imputer=='mean':
        imputer = SimpleImputer(strategy='mean')
    elif imputer=='median':
        imputer = SimpleImputer(strategy='median')
    elif imputer=='most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    else:
        raise ValueError("Imputer phải là 'knn', 'mean', 'median' hoặc 'most_frequent'")

    # Scaler
    if scaler == 'minmax':
        scaler_transformer = MinMaxScaler(feature_range)
    elif scaler == 'standard':
        scaler_transformer = StandardScaler()
    else:
        raise ValueError("Scaler phải là 'minmax' hoặc 'standard'")

    numeric_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler_transformer)
    ])

    #encoder
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feature),
            ('cat', categorical_transformer, cat_feature)
        ]
    )

# ------------------------------
# 3. Tạo pipeline tổng
# ------------------------------

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components)),
        ('model', model)
    ])

# ------------------------------
# 4. Train + đánh giá
# ------------------------------
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)

    # lấy tỷ lệ phương sai giữ lại
    explained_var = pipe.named_steps['pca'].explained_variance_ratio_
    total_var = explained_var.sum()

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "F1-score (macro)": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "Explained Variance (%)": round(total_var * 100, 2)
    }

    return metrics, pipe
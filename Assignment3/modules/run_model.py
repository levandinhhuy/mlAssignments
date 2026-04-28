from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import sparse

def run_models(
    Xtr, ytr, Xva, yva, Xte, yte,
    model_params,
    print_reports: bool = True,
    pca: bool = False,
    normalize: bool = False
):
    """
    Train & chọn model tốt nhất theo VAL accuracy, rồi đánh giá trên TEST.

    model_params: dict tên_mô_hình -> dict tham số
      Hỗ trợ các key:
        - 'RandomForest' -> RandomForestClassifier(**params)
        - 'LogisticRegression' -> LogisticRegression(**params)
        - 'LinearSVC' -> LinearSVC(**params)
        - 'SVC' -> SVC(**params)   # đặt probability=True nếu cần predict_proba

    Return:
      best_name, best_model, results
        results[model_name] = {'model': model, 'val_acc': float, 'test_acc': float}
    """
    results = {}

    # 1) RandomForest
    if 'RandomForest' in model_params:
        rf_params = model_params['RandomForest']
        if normalize:
            rf = make_pipeline(Normalizer(norm= "l2"),
                               RandomForestClassifier(**rf_params,
                                                      n_jobs= -1, random_state= 42))
        else:
            rf = RandomForestClassifier(**rf_params, n_jobs= -1, random_state= 42)

        rf.fit(Xtr, ytr)
        yva_pred = rf.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[RandomForest] VAL acc = {val_acc}")
        results['RandomForest'] = {'model': rf, 'val_acc': val_acc}

    # 2) Logistic Regression (nên scale)
    if 'LogisticRegression' in model_params:
        lr_params = model_params['LogisticRegression']
        if normalize and pca:
            lr = make_pipeline(StandardScaler(),Normalizer(norm="l2"),
                                                PCA(n_components=0.95,
                                                    whiten=True,random_state=42),
                                                LogisticRegression(**lr_params))
        elif normalize and ( not pca):
            lr = make_pipeline(Normalizer(norm="l2"), LogisticRegression(**lr_params))
        elif (not normalize) and pca:
            lr = make_pipeline(StandardScaler(),PCA(n_components=0.95,
                                                    whiten=True,random_state=42),
                                               LogisticRegression(**lr_params))
        else:
            lr = make_pipeline(StandardScaler(), LogisticRegression(**lr_params))
        lr.fit(Xtr, ytr)
        yva_pred = lr.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[LogisticRegression] VAL acc = {val_acc}")
        results['LogisticRegression'] = {'model': lr, 'val_acc': val_acc}

    # 3) LinearSVC (nhanh, mạnh; không có predict_proba)
    if 'LinearSVC' in model_params:
        lsvm_params = model_params['LinearSVC']
        if normalize and pca:
            lsvm = make_pipeline(StandardScaler(),Normalizer(norm="l2"),
                                                PCA(n_components=0.95,
                                                    whiten=True,random_state=42),
                                                LinearSVC(**lsvm_params))
        elif normalize and ( not pca):
            lsvm = make_pipeline(Normalizer(norm="l2"), LinearSVC(**lsvm_params))
        elif (not normalize) and pca:
            lsvm = make_pipeline(StandardScaler(),PCA(n_components=0.95,
                                                    whiten=True,random_state=42),
                                               LinearSVC(**lsvm_params))
        else:
            lsvm = make_pipeline(StandardScaler(), LinearSVC(**lsvm_params))

        lsvm.fit(Xtr, ytr)
        yva_pred = lsvm.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[LinearSVC] VAL acc = {val_acc}")
        results['LinearSVC'] = {'model': lsvm, 'val_acc': val_acc}

    # 4) SVC (có xác suất nếu probability=True; chậm hơn LinearSVC)
    if 'SVC' in model_params:
        svc_params = model_params['SVC']
        if normalize and pca:
            svc = make_pipeline(StandardScaler(),Normalizer(norm="l2"),
                                                PCA(n_components=0.95,
                                                    whiten=True,random_state=42),
                                                SVC(**svc_params))
        elif normalize and ( not pca):
            svc = make_pipeline(Normalizer(norm="l2"), SVC(**svc_params))
        elif (not normalize) and pca:
            svc = make_pipeline(StandardScaler(),PCA(n_components=0.95,
                                                    whiten=True,random_state=42),
                                               SVC(**svc_params))
        else:
            svc = make_pipeline(StandardScaler(), SVC(**svc_params))
        svc.fit(Xtr, ytr)
        yva_pred = svc.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[SVC] VAL acc = {val_acc}")
        results['SVC'] = {'model': svc, 'val_acc': val_acc}

    if not results:
        raise ValueError("Không có mô hình nào để chạy. Hãy truyền model_params hợp lệ.")

    return results

def evaluate_model_on_test(model, Xte, yte, model_name):
    """Evaluates a trained model on the test set and prints the report."""
    print(f"\n--- Đánh giá mô hình {model_name} tốt nhất trên tập Test ---")
    yte_pred = model.predict(Xte)
    test_accuracy = accuracy_score(yte, yte_pred)

    print(f"Test Accuracy: {test_accuracy}")
    print(classification_report(yte, yte_pred, zero_division=0))
    return test_accuracy
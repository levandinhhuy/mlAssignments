import os, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse

########################################################################################################
#                                                                                                      #
#                           Pipeline trích xuất đặc trưng bằng BoW và TF-IDF                           #
#                                                                                                      #
########################################################################################################    
                                    # --------- BoW ----------
def BoW_extractor(train, val, test, max_features = 5000, ngram_range = (1,2), min_df = 2, max_df = 0.9):
    bow_vectorizer = CountVectorizer(
        max_features= max_features, #Chọn ra số lượng đặc trưng phổ biến nhất 
        ngram_range=ngram_range,    #Cụm từ đơn lẻ hoặc cụm từ
        min_df=min_df,              #Chỉ chọn các từ hoặc n-gram xuất hiện ít nhất trong min_df tài liệu
        max_df=max_df,              #Không chọn các từ hoặc n-gram xuất hiện trong max_df (90%) tài liệu.
        lowercase=False,            #text đã được lowercase ở bước preprocessing
        stop_words = None,          #Đã được loại bỏ domain stopword và các stopword trong thư viện nlkt
        tokenizer=None,             #Đã được tokenize
        preprocessor=None,          #Đã tiền xử lý
        analyzer="word"             #Giữ mặc định word-level
    )
    Xtr_bow = bow_vectorizer.fit_transform(train)
    Xva_bow = bow_vectorizer.transform(val)
    Xte_bow = bow_vectorizer.transform(test)
    return Xtr_bow, Xva_bow, Xte_bow, bow_vectorizer


                                    # --------- TF - IDF ----------
def TFIDF_extractor(train, val, test, max_features = 5000, ngram_range = (1,2), min_df = 2, max_df = 0.9):
    tfidf = TfidfVectorizer(
        max_features=max_features,  #Chọn ra số lượng đặc trưng phổ biến nhất 
        ngram_range=ngram_range,    #Cụm từ đơn lẻ hoặc cụm từ
        min_df= min_df,             #Chỉ chọn các từ hoặc n-gram xuất hiện ít nhất trong min_df tài liệu
        max_df = max_df,            #Không chọn các từ hoặc n-gram xuất hiện trong max_df (90%) tài liệu.
        lowercase=False,            #text đã được lowercase ở bước preprocessing
        stop_words = None,          #Đã được loại bỏ domain stopword và các stopword trong thư viện nlkt
        preprocessor=None,          #Đã được tiền xử lý
        tokenizer=None,             #Đã được tokenize
        use_idf=True,               #Tần suất từ sẽ được nhân với trọng số IDF
        sublinear_tf=True,          #Sử dụng một hàm logarit cho giá trị TF
        analyzer="word"             #Giữ mặc định word-level
    )
    Xtr_tfidf = tfidf.fit_transform(train)
    Xva_tfidf = tfidf.transform(val)
    Xte_tfidf = tfidf.transform(test)
    return Xtr_tfidf, Xva_tfidf, Xte_tfidf, tfidf

########################################################################################################
#                                                                                                      #
#                           Các hàm phụ trích xuất báo cáo vectorizer                                  #
#                                                                                                      #
########################################################################################################  
def _sparse_stats(X):
    n_rows, n_cols = X.shape
    nnz = X.nnz
    density = nnz / (n_rows * n_cols) if n_rows*n_cols else 0.0

    return {
        "shape": (n_rows, n_cols),
        "nnz": int(nnz),
        "density": float(density),
    }

#--------Hàm lấy các tham số trong vectorizer---------
def _vectorizer_params(vec):
    params = {
        "type": vec.__class__.__name__,
        "ngram_range": getattr(vec, "ngram_range", None),
        "min_df": getattr(vec, "min_df", None),
        "max_df": getattr(vec, "max_df", None),
        "max_features": getattr(vec, "max_features", None),
        "lowercase": getattr(vec, "lowercase", None),
        "token_pattern": getattr(vec, "token_pattern", None),
        "use_idf": getattr(vec, "use_idf", None) if hasattr(vec, "use_idf") else None,
        "sublinear_tf": getattr(vec, "sublinear_tf", None) if hasattr(vec, "sublinear_tf") else None,
        "stop_words": "custom" if getattr(vec, "stop_words", None) not in (None, "english") else getattr(vec, "stop_words", None),
    }
    vocab_size = len(vec.get_feature_names_out())
    params["vocab_size"] = int(vocab_size)
    return params

#-------- Hàm ghi báo cáo các vectorizer------------
def report_vectorizer(vec, Xtr, Xva, Xte, title=None):
    lines = []
    if title:
        bar = "=" * len(title)
        lines += [bar, title, bar, ""]

    params = _vectorizer_params(vec)
    lines.append(">>> Vectorizer params:")
    for k, v in params.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")  

    lines.append(">>> Matrix stats:")
    for name, M in [("Train", Xtr), ("Val", Xva), ("Test", Xte)]:
        s = _sparse_stats(M)
        lines.append(f"  [{name}] shape={s['shape']}  nnz={s['nnz']:,}  density={s['density']:.6f}")

    return "\n".join(lines)

########################################################################################################
#                                                                                                      #
#                           Pipeline chính để trích xuất đặc trưng bằng BoW và TF-IDF                  #
#                                                                                                      #
########################################################################################################  
def features_extractor_classic(train, val, test, mode = "BoW", **kwargs):
    if mode == "BoW":
        Xtr, Xva, Xte, vec = BoW_extractor(train, val, test, **kwargs)
        outdir = "features/bow"
        res = report_vectorizer(vec, Xtr, Xva, Xte, title="Bag-of-Words Report")
    elif mode == "TFIDF":
        Xtr, Xva, Xte, vec = TFIDF_extractor(train, val, test, **kwargs)
        outdir = "features/tfidf"
        res = report_vectorizer(vec, Xtr, Xva, Xte, title="TF-IDF Report")
    else:
        raise ValueError ("mode phải là BoW hoặc TFIDF")
    os.makedirs(outdir, exist_ok=True)
    sparse.save_npz(f"{outdir}/Xtr.npz", Xtr)
    sparse.save_npz(f"{outdir}/Xva.npz", Xva)
    sparse.save_npz(f"{outdir}/Xte.npz", Xte)
    return res
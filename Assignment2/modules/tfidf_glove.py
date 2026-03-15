import os
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

#--------Hàm tách thành các token--------
def to_tokens(texts):
    return [t.split() for t in texts]

#--------Hàm tải mô hình embended glove từ trên web và load w2v format--------
def load_glove_model():
    glove_zip = "glove.2024.wikigiga.300d.zip"
    glove_dir = "glove.2024.300d"
    glove_input = f"{glove_dir}/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
    w2v_output = "glove.2024.300d.w2v.txt"

    if not os.path.exists(w2v_output):
        glove2word2vec(glove_input, w2v_output)

    wv = KeyedVectors.load_word2vec_format(w2v_output, binary=False)
    return wv

#------Xây dựng IDF_map-----------
def build_tfidf(train_texts):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.9,
        lowercase=False,
        sublinear_tf=True,
        token_pattern=r"[a-zA-Z]+"
    )
    tfidf.fit(train_texts)
    idf_map = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    idf_default = np.median(list(idf_map.values()))
    return idf_map, idf_default

#----Tạo vector tài liệu bằng trung bình trọng số (Tf x IDF)---------
def sent_vec_tfidf(tokens, keyedvecs, idf_map, idf_default):
    dim = keyedvecs.vector_size
    cnt = Counter(tokens)
    wsum = np.zeros(dim, dtype="float32")
    wtot = 0.0
    for t, tf in cnt.items():
        if t in keyedvecs:
            idf = idf_map.get(t, idf_default)
            wt = tf * idf
            wsum += keyedvecs[t] * wt
            wtot += wt
    return (wsum / wtot) if wtot > 0 else np.zeros(dim, dtype="float32")

#-------Chuyển đổi tập tài liệu thành ma trận Numpy-------
def docs_to_matrix(token_lists, keyedvecs, idf_map, idf_default):
    return np.vstack([sent_vec_tfidf(toks, keyedvecs, idf_map, idf_default) for toks in token_lists])


#--------Pipepline chính để chạy trích xuất đặc trưng bằng phương pháp Tf-idf kết hợp glove----
def run_tfidf_glove(tr_clean, va_clean, te_clean, output_dir="features/tfidf_glove"):

    Xtr_tok = to_tokens(tr_clean)
    Xva_tok = to_tokens(va_clean)
    Xte_tok = to_tokens(te_clean)

    wv = load_glove_model()

    idf_map, idf_default = build_tfidf(tr_clean)

    Xtr_w2v = docs_to_matrix(Xtr_tok, wv, idf_map, idf_default)
    Xva_w2v = docs_to_matrix(Xva_tok, wv, idf_map, idf_default)
    Xte_w2v = docs_to_matrix(Xte_tok, wv, idf_map, idf_default)

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/Xtr_w2v.npy", Xtr_w2v.astype(np.float32))
    np.save(f"{output_dir}/Xva_w2v.npy", Xva_w2v.astype(np.float32))
    np.save(f"{output_dir}/Xte_w2v.npy", Xte_w2v.astype(np.float32))

    return Xtr_w2v, Xva_w2v, Xte_w2v
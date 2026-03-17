import tensorflow as tf
import numpy as np
import os

def run_extraction(model_name, data_pipeline, output_dir):
    """
    Trích xuất đặc trưng từ một mô hình (ResNet50, VGG16, EfficientNetB0) 
    và lưu ra file .npy.

    Args:
        model_name (str): Tên mô hình ('resnet50', 'vgg16', 'efficientnetb0').
        data_pipeline (dict): Dict chứa {'train': train_ds, 'val': val_ds, 'test': test_ds}.
        output_dir (str): Đường dẫn thư mục để lưu kết quả .npy.
    """
    
    print(f"--- Bắt đầu trích xuất cho: {model_name.upper()} ---")

    # 3. Tải và cấu hình mô hình extractor
    if model_name == "resnet50":
        # Tải base_model ResNet50
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3) # Input phải khớp với lúc tạo pipeline
        )
        base_model.trainable = False
        
        # Lắp ráp extractor (thêm GAP)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        extractor = tf.keras.Sequential([base_model, global_average_layer], name="resnet50_extractor")
        print("Đã tải và lắp ráp mô hình ResNet50.")

    elif model_name == "vgg16":
        # Tải base_model VGG16
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Lắp ráp extractor (thêm GAP)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        extractor = tf.keras.Sequential([base_model, global_average_layer], name="vgg16_extractor")
        print("Đã tải và lắp ráp mô hình VGG16.")

    elif model_name == "efficientnetb0":
        # Tải base_model EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Lắp ráp extractor (thêm GAP)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        extractor = tf.keras.Sequential([base_model, global_average_layer], name="efficientnetb0_extractor")
        print("Đã tải và lắp ráp mô hình EfficientNetB0.")
    
    else:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ.")

    # 4. Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    print(f"Đã tạo/kiểm tra thư mục: {output_dir}")

    # 5. Trích xuất (lặp qua train/val/test)
    for split_name, dataset in data_pipeline.items():
        if dataset is None:
            print(f"Bỏ qua tập {split_name} (không có dữ liệu).")
            continue
            
        print(f"Đang trích xuất tập {split_name}...")
        
        features_list = []
        labels_list = []

        # Lặp qua từng batch
        for image_batch, label_batch in dataset:
            # Chạy 'predict' (inference mode)
            features_batch = extractor(image_batch, training=False) 
            features_list.append(features_batch)
            labels_list.append(label_batch)

        # Ghép nối các batch
        X_features = np.concatenate(features_list)
        y_labels = np.concatenate(labels_list)
        
        print(f"...Đã xong tập {split_name}. Shape: {X_features.shape}")

        # 6. Lưu file .npy
        np.save(os.path.join(output_dir, f'X_{split_name}.npy'), X_features)
        np.save(os.path.join(output_dir, f'y_{split_name}.npy'), y_labels)
        print(f"Đã lưu file X_{split_name}.npy và y_{split_name}.npy")

    print(f"--- HOÀN THÀNH: {model_name.upper()} ---")

    
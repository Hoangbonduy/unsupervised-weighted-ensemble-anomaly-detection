# train_meta_model_inception_v2.py

import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import lớp InceptionTimeRegressor từ file chúng ta vừa tạo
from inception_regressor import InceptionTimeRegressor

# --- PHẦN 1: HÀM THỰC THI CHÍNH ---

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print("="*80)
    print("Bắt đầu Giai đoạn 3: Huấn luyện Meta-Model với InceptionTime (Kiến trúc từ tác giả)")
    print("="*80)

    # 1. Định nghĩa các file và thư mục
    meta_feature_file = 'results_enhanced_2/meta_feature_matrix_M.csv'
    performance_file = 'results_enhanced_2/performance_matrix_P.csv'
    output_dir = 'inception_model_output/'
    metadata_file = 'results_enhanced_2/meta_model_metadata.joblib'

    os.makedirs(output_dir, exist_ok=True)

    # 2. Tải và chuẩn bị dữ liệu
    print("Đang tải và chuẩn bị ma trận M và P...")
    df_M = pd.read_csv(meta_feature_file)
    df_P = pd.read_csv(performance_file)

    # Căn chỉnh dữ liệu
    df_M = df_M.sort_values(by='place_id').reset_index(drop=True)
    df_P = df_P.sort_values(by='place_id').reset_index(drop=True)

    if not df_M['place_id'].equals(df_P['place_id']):
        print("LỖI: 'place_id' trong hai file không khớp nhau.")
        return

    X = df_M.drop(columns=['place_id']).values
    y = df_P.drop(columns=['place_id']).values
    
    # Kiểm tra và xử lý dữ liệu NaN/Inf
    print(f"Kiểm tra dữ liệu X: NaN={np.isnan(X).sum()}, Inf={np.isinf(X).sum()}")
    print(f"Kiểm tra dữ liệu y: NaN={np.isnan(y).sum()}, Inf={np.isinf(y).sum()}")
    
    # Thay thế NaN và Inf bằng 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Chuẩn hóa dữ liệu đầu vào X về khoảng [0, 1]
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    
    # Chuẩn hóa dữ liệu đầu ra y về khoảng [0, 1] 
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y)
    
    # Reshape dữ liệu đầu vào cho InceptionTime: (số mẫu, số đặc trưng, 1)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    print(f"\nChuẩn bị dữ liệu hoàn tất:")
    print(f"  - Kích thước Ma trận M (đặc trưng) sau reshape: {X.shape}")
    print(f"  - Kích thước Ma trận P (mục tiêu): {y.shape}")
    print(f"  - Phạm vi X: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  - Phạm vi y: [{y.min():.3f}, {y.max():.3f}]")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  - Dữ liệu được chia: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra.")

    # 4. Khởi tạo và Huấn luyện Meta-Model
    input_shape = X_train.shape[1:]
    output_size = y_train.shape[1]

    print("\nBắt đầu khởi tạo mô hình InceptionTimeRegressor...")
    print("Cấu hình Early Stopping: Dừng sau 10 epochs không cải thiện val_loss")
    
    # Sử dụng lớp InceptionTimeRegressor đã được tùy chỉnh
    regressor = InceptionTimeRegressor(
        output_directory=output_dir,
        input_shape=input_shape,
        output_size=output_size,
        depth=6,             # Giữ nguyên các siêu tham số mặc định của tác giả
        kernel_size=41,
        nb_filters=32,
        batch_size=16,       # Giảm batch size cho tập dữ liệu nhỏ
        nb_epochs=500,       # Tăng epochs, EarlyStopping sẽ dừng khi cần
        verbose=True
    )
    
    # Huấn luyện mô hình
    regressor.fit(X_train, y_train, X_test, y_test)
    
    print("--- HUẤN LUYỆN HOÀN TẤT ---")

    # 5. Đánh giá mô hình trên tập test
    print("\nĐang đánh giá hiệu suất của model tốt nhất trên tập kiểm tra...")
    y_pred = regressor.model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"  - Lỗi Trung bình Bình phương (MSE): {mse:.4f}")
    print(f"  - Căn bậc hai MSE (RMSE): {np.sqrt(mse):.4f}")

    # 6. Lưu metadata và scalers
    metadata = {
        "meta_feature_names": df_M.drop(columns=['place_id']).columns.tolist(),
        "performance_model_names": df_P.drop(columns=['place_id']).columns.tolist(),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    }
    joblib.dump(metadata, metadata_file)
    print(f"\nMetadata và scalers đã được lưu vào file: '{metadata_file}'")
    print(f"Model tốt nhất đã được lưu trong thư mục: '{output_dir}'")
    print("--- QUY TRÌNH HOÀN TẤT ---")

if __name__ == '__main__':
    main()
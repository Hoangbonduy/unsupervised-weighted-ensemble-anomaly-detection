# train_meta_model.py

import pandas as pd
import numpy as np
import joblib
import os

# --- PHẦN 0: IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
except ImportError:
    print("LỖI: Thư viện 'scikit-learn' chưa được cài đặt.")
    print("Vui lòng chạy lệnh: pip install scikit-learn")
    exit()

# --- PHẦN 1: HÀM THỰC THI CHÍNH ---

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print("Bắt đầu Giai đoạn 3: Huấn luyện Meta-Model")

    # 1. Định nghĩa tên file đầu vào và file model đầu ra
    meta_feature_file = 'results_enhanced/meta_feature_matrix_M.csv'
    performance_file = 'results_enhanced/performance_matrix_P.csv'
    output_model_file = 'results_enhanced/meta_model.joblib'

    # Kiểm tra sự tồn tại của các file đầu vào
    if not os.path.exists(meta_feature_file) or not os.path.exists(performance_file):
        print(f"LỖI: Không tìm thấy file đầu vào.")
        print(f"  - Cần file '{meta_feature_file}' (từ bước 2)")
        print(f"  - Cần file '{performance_file}' (từ bước 1)")
        print("Vui lòng chạy các kịch bản trước để tạo ra các file này.")
        return

    # 2. Tải Ma trận Meta-Feature (M) và Ma trận Hiệu suất (P)
    print("Đang tải các ma trận M và P...")
    try:
        df_M = pd.read_csv(meta_feature_file)
        df_P = pd.read_csv(performance_file)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    # 3. Căn chỉnh và chuẩn bị dữ liệu
    print("Đang căn chỉnh dữ liệu giữa M và P...")
    # Sắp xếp cả hai DataFrame theo 'place_id' để đảm bảo các hàng khớp với nhau
    df_M = df_M.sort_values(by='place_id').reset_index(drop=True)
    df_P = df_P.sort_values(by='place_id').reset_index(drop=True)

    # Kiểm tra xem các place_id có khớp nhau không
    if not df_M['place_id'].equals(df_P['place_id']):
        print("LỖI: 'place_id' trong hai file không khớp nhau. Không thể huấn luyện.")
        # Tìm các id không khớp để debug
        m_ids = set(df_M['place_id'])
        p_ids = set(df_P['place_id'])
        print(f"  - Các ID chỉ có trong M: {m_ids - p_ids}")
        print(f"  - Các ID chỉ có trong P: {p_ids - m_ids}")
        return

    # Tách các đặc trưng (X) và mục tiêu (y)
    # X là tất cả các cột trong M trừ 'place_id'
    X = df_M.drop(columns=['place_id'])
    
    # y là tất cả các cột trong P trừ 'place_id'
    y = df_P.drop(columns=['place_id'])

    print("\nChuẩn bị dữ liệu hoàn tất:")
    print(f"  - Kích thước Ma trận M (đặc trưng): {X.shape}")
    print(f"  - Kích thước Ma trận P (mục tiêu):  {y.shape}")

    # (Tùy chọn) Chia dữ liệu để đánh giá mô hình
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  - Dữ liệu được chia: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra.")

    # 4. Khởi tạo và Huấn luyện Meta-Model
    print("\nBắt đầu huấn luyện Meta-Model (Multi-Output Random Forest Regressor)...")
    
    # Định nghĩa mô hình cơ sở (RandomForestRegressor)
    # n_jobs=-1 để sử dụng tất cả các nhân CPU, giúp tăng tốc độ huấn luyện
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Bọc mô hình cơ sở bằng MultiOutputRegressor để nó có thể dự đoán nhiều đầu ra
    meta_model = MultiOutputRegressor(estimator=base_regressor)
    
    # Huấn luyện mô hình trên toàn bộ dữ liệu
    meta_model.fit(X_train, y_train)
    
    print("--- HUẤN LUYỆN HOÀN TẤT ---")

    # 5. Đánh giá mô hình (tùy chọn nhưng nên có)
    print("\nĐang đánh giá hiệu suất của Meta-Model trên tập kiểm tra...")
    y_pred = meta_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"  - Lỗi Trung bình Bình phương (MSE): {mse:.4f}")
    print(f"  - Căn bậc hai MSE (RMSE): {np.sqrt(mse):.4f}")
    
    # Hiển thị ví dụ về dự đoán
    print("\nVí dụ về dự đoán so với thực tế (cho 5 mẫu đầu tiên của tập test):")
    comparison_df = pd.DataFrame(y_test.values, columns=[f'Thực tế_{col}' for col in y.columns])
    pred_df = pd.DataFrame(y_pred, columns=[f'Dự đoán_{col}' for col in y.columns])
    print(pd.concat([comparison_df.head(), pred_df.head()], axis=1))


    # 6. Lưu lại model đã huấn luyện
    print(f"\nĐang lưu Meta-Model đã huấn luyện vào file: '{output_model_file}'...")
    joblib.dump(meta_model, output_model_file)
    
    # (Tùy chọn) Lưu cả tên cột để tham chiếu sau này
    metadata = {
        "meta_feature_names": X.columns.tolist(),
        "performance_model_names": y.columns.tolist()
    }
    joblib.dump(metadata, 'results_enhanced/meta_model_metadata.joblib')

    print("--- QUY TRÌNH HOÀN TẤT ---")

if __name__ == '__main__':
    main()
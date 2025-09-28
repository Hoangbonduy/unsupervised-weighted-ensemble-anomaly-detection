# 3_train_inception_regressor_rl_top_k.py
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error

# Sử dụng lại lớp InceptionTimeRegressor gốc
from inception_regressor_rl import InceptionTimeRegressor

def get_top_k_ensemble_performance(y_true, y_pred_scores, k):
    """
    Hàm tính hiệu suất ensemble bằng cách chỉ sử dụng K mô hình tốt nhất.
    - y_true: Điểm VUS-ROC thực tế.
    - y_pred_scores: Điểm VUS-ROC dự đoán bởi InceptionTime.
    - k: Số lượng mô hình tốt nhất để chọn.
    """
    all_weighted_performances = []
    
    # Duyệt qua từng mẫu trong tập test
    for i in range(len(y_pred_scores)):
        scores = y_pred_scores[i]
        true_vus = y_true[i]
        
        # Lấy chỉ số của K mô hình có điểm dự đoán cao nhất
        top_k_indices = np.argsort(scores)[-k:]
        
        # Lấy điểm dự đoán của K mô hình này
        top_k_scores = scores[top_k_indices]
        
        # Chuẩn hóa lại trọng số của K mô hình này để tổng bằng 1 (giống softmax)
        # Điều này đảm bảo mô hình có điểm cao hơn sẽ có trọng số lớn hơn
        weights = np.exp(top_k_scores) / np.sum(np.exp(top_k_scores))
        
        # Lấy điểm VUS-ROC thực tế của K mô hình này
        true_top_k_vus = true_vus[top_k_indices]
        
        # Tính hiệu suất có trọng số cho mẫu này
        weighted_performance = np.sum(true_top_k_vus * weights)
        all_weighted_performances.append(weighted_performance)
        
    # Trả về hiệu suất trung bình trên toàn bộ tập test
    return np.mean(all_weighted_performances)

def main():
    print("="*80)
    print("Bắt đầu: Huấn luyện InceptionTimeRegressor với Learning Rate Scheduler")
    print("="*80)

    # --- 1. Định nghĩa các file và thư mục ---
    ts_file = 'data/cleaned_data_after_idx30.csv'
    performance_file = 'results_enhanced_final/performance_matrix_P.csv'
    output_dir = 'inception_regressor_rl_output_top_k_final/' # Thư mục mới

    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Tải và xử lý chuỗi thời gian thô (X) ---
    print("--> Bước 1: Đang tải và xử lý chuỗi thời gian thô...")
    df_ts = pd.read_csv(ts_file)
    place_ids_to_drop = [4701454798439365089, 4711168091885605243]
    df_ts = df_ts[~df_ts['placeId'].isin(place_ids_to_drop)]
    rename_dict = {'placeId': 'place_id', 'view': 'value'}
    df_ts = df_ts.rename(columns=rename_dict)
    df_ts['date'] = pd.to_datetime(df_ts['date'])
    df_ts = df_ts.sort_values(by=['place_id', 'date'])
    ts_groups = df_ts.groupby('place_id')['value'].apply(list)
    X_padded = pad_sequences(ts_groups, padding='post', dtype='float32')
    X = X_padded[:, :, np.newaxis]
    for i in range(X.shape[0]):
        scaler_x = StandardScaler()
        X[i] = scaler_x.fit_transform(X[i])
    print(f"Hoàn thành xử lý X. Kích thước cuối cùng: {X.shape}")

    # --- 3. Tải và xử lý Target (y) ---
    print("\n--> Bước 2: Đang tải và xử lý Target VUS-ROC...")
    df_P = pd.read_csv(performance_file)
    if 'placeId' in df_P.columns:
        df_P = df_P.rename(columns={'placeId': 'place_id'})
    df_P = df_P.set_index('place_id').loc[ts_groups.index].reset_index()
    y_raw = df_P.drop(columns=['place_id']).values
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y_raw)
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.joblib'))
    print(f"Hoàn thành xử lý y. Kích thước cuối cùng: {y.shape}")

    # --- 4. Chia dữ liệu và huấn luyện ---
    print("\n--> Bước 3: Chia dữ liệu và bắt đầu huấn luyện...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]
    output_size = y_train.shape[1]

    print("\n🔧 Cấu hình Training:")
    print("  ✅ Early Stopping: patience=50")
    print("  ✅ Learning Rate Scheduler: ReduceLROnPlateau")
    print("  ✅ Initial LR: 0.001, giảm 50% sau 10 epochs không cải thiện")
    print("  ✅ Model Checkpoint: Lưu model tốt nhất")
    print("  ✅ Minimum LR: 1e-7")

    regressor = InceptionTimeRegressor(
        output_directory=output_dir,
        input_shape=input_shape,
        output_size=output_size,
        batch_size=32,
        nb_epochs=300,
        # Thêm patience để có Early Stopping hiệu quả
        patience=40,
        # Thêm learning rate scheduler
        initial_lr=0.001,
        verbose=True
    )
    regressor.fit(X_train, y_train, X_test, y_test)
    
    # --- 5. Đánh giá với chiến lược Top-K ---
    print("\n--> Bước 4: Đánh giá mô hình trên tập test với chiến lược Top-K...")
    y_pred_scaled = regressor.model.predict(X_test)
    
    # Chuyển đổi ngược lại thang đo gốc
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # Tính hiệu suất ensemble baseline (trung bình cộng)
    baseline_performance = np.mean(np.mean(y_test_orig, axis=1))
    
    # Tính hiệu suất với Top-K=3
    top_3_perf = get_top_k_ensemble_performance(y_test_orig, y_pred, k=3)
    
    # Tính hiệu suất với Top-K=4
    top_4_perf = get_top_k_ensemble_performance(y_test_orig, y_pred, k=4)

    print("\n--- Kết quả đánh giá Ensemble trên tập Test ---")
    print(f" - Hiệu suất VUS-ROC của Baseline (trung bình cộng): {baseline_performance:.4f}")
    print(f" - Hiệu suất VUS-ROC của Ensemble Top-3: {top_3_perf:.4f}")
    print(f" - Hiệu suất VUS-ROC của Ensemble Top-4: {top_4_perf:.4f}")
    
    print("\n--- Phân tích MSE (chỉ để tham khảo) ---")
    mse = mean_squared_error(y_test_orig, y_pred)
    print(f" - Lỗi Trung bình Bình phương (MSE) dự đoán VUS-ROC: {mse:.4f}")

    print(f"\n--- QUY TRÌNH HOÀN TẤT ---")
    print(f"Model tốt nhất đã được lưu trong thư mục: '{output_dir}'")
    
    # Tạo biểu đồ so sánh performance
    create_performance_comparison_plot(baseline_performance, top_3_perf, top_4_perf, output_dir)

def create_performance_comparison_plot(baseline_perf, top3_perf, top4_perf, output_dir):
    """
    Tạo biểu đồ so sánh performance giữa các phương pháp ensemble
    """
    methods = ['Baseline\n(Average)', 'Top-3\nEnsemble', 'Top-4\nEnsemble']
    performances = [baseline_perf, top3_perf, top4_perf]
    colors = ['lightcoral', 'skyblue', 'lightgreen']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, performances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Thêm giá trị trên mỗi cột
    for bar, perf in zip(bars, performances):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{perf:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('So sánh Performance VUS-ROC của các phương pháp Ensemble', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('VUS-ROC Score', fontsize=12, fontweight='bold')
    plt.xlabel('Phương pháp Ensemble', fontsize=12, fontweight='bold')
    
    # Cải thiện hiển thị
    plt.ylim(0, max(performances) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Lưu biểu đồ
    plot_path = os.path.join(output_dir, 'ensemble_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'ensemble_performance_comparison.pdf'), bbox_inches='tight')
    
    print(f"\n📊 Biểu đồ so sánh Performance đã được lưu: {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
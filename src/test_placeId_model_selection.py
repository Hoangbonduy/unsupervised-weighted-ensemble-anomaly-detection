# ==============================================================================
# test_and_visualize_single_id.py
#
# Script này thực hiện toàn bộ quy trình cho MỘT placeId duy nhất:
# 1. Đồng bộ dữ liệu và nhãn.
# 2. Lựa chọn mô hình tốt nhất bằng phương pháp không giám sát.
# 3. Chạy mô hình đã chọn để tạo dự đoán.
# 4. Tính F1-score so với nhãn thật.
# 5. Trực quan hóa kết quả trên biểu đồ.
# ==============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import warnings
import main
import anomaly_detection_base_model

# --- 0. CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT ---
# !pip install pandas numpy scikit-learn scipy tsaug matplotlib pyod sranodec statsmodels
from pyod.models.hbos import HBOS
import sranodec as anom
from statsmodels.tsa.seasonal import seasonal_decompose # Dùng để thay thế hàm cũ

warnings.filterwarnings('ignore')

# ==============================================================================
# PHẦN B: KỊCH BẢN CHÍNH
# ==============================================================================
if __name__ == '__main__':
    # --- 1. CẤU HÌNH ---
    # !!! THAY ĐỔI CÁC GIÁ TRỊ NÀY CHO PHÙ HỢP VỚI BẠN !!!
    TEST_PLACE_ID = 4612249398246946328  # Chọn một placeId để kiểm tra (đổi thành int)
    # Đường dẫn tương đối từ thư mục gốc dự án (cwd khi chạy thường là root repo)
    DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'labels'
    CONTAMINATION_RATE_ESTIMATE = 0.02 # Ước tính tỷ lệ bất thường chung (2%)

    # --- 2. ĐỌC VÀ ĐỒNG BỘ HÓA DỮ LIỆU CHO MỘT PLACEID ---
    print(f"--- Bắt đầu xử lý cho PlaceID: {TEST_PLACE_ID} ---")
    
    # Đọc dữ liệu nguồn
    data_full = pd.read_csv(DATA_FILE_PATH)
    # Chuẩn hóa placeId để tránh lỗi do số nguyên lớn bị đọc thành float/scientific
    if 'placeId' not in data_full.columns:
        raise RuntimeError("Cột 'placeId' không tồn tại trong dữ liệu.")
    data_full['placeId_norm'] = pd.to_numeric(data_full['placeId'], errors='coerce').astype('Int64')
    TEST_PLACE_ID_INT = np.int64(TEST_PLACE_ID)
    ts_source = data_full[data_full['placeId_norm'] == TEST_PLACE_ID_INT].copy()
    
    print(f"Tìm thấy {len(ts_source)} records trong dữ liệu chính cho PlaceID {TEST_PLACE_ID}")
    if ts_source.empty:
        # Gợi ý debug nhanh
        unique_sample = data_full['placeId'].drop_duplicates().head(5).tolist()
        print("Không tìm thấy dữ liệu cho placeId yêu cầu. Một vài placeId có trong file:", unique_sample)
        raise SystemExit(1)
    
    # Đọc dữ liệu nhãn
    label_path = os.path.join(LABELS_DIR, f"label_{TEST_PLACE_ID}.csv")
    if not os.path.exists(label_path):
        print(f"Không tìm thấy file label: {label_path}")
        # Thử đường dẫn thay thế nếu chạy từ src/
        alt_label_path = os.path.join('..', 'labels', f'label_{TEST_PLACE_ID}.csv')
        if os.path.exists(alt_label_path):
            print(f"Thử dùng đường dẫn thay thế: {alt_label_path}")
            label_path = alt_label_path
        else:
            raise FileNotFoundError(f"Thiếu file nhãn cho PlaceID {TEST_PLACE_ID}. Hãy kiểm tra thư mục 'labels/'.")
    df_label = pd.read_csv(label_path)
    
    print(f"Tìm thấy {len(df_label)} records trong file label")
    
    # Đảm bảo cột date có cùng format
    ts_source['date'] = ts_source['date'].astype(str)
    df_label['date'] = df_label['date'].astype(str)
    
    # Debug: kiểm tra một vài dates
    print("Một vài dates từ dữ liệu chính:", ts_source['date'].head().tolist())
    print("Một vài dates từ file label:", df_label['date'].head().tolist())
    
    # Đồng bộ hóa bằng inner merge
    merged_df = pd.merge(ts_source, df_label, on='date', how='inner', suffixes=('_main', '_label'))
    
    print(f"Sau khi merge: {len(merged_df)} records")
    
    if merged_df.empty:
        print("LỖI: Dữ liệu và nhãn không có ngày chung. Dừng lại.")
        exit()
        
    print(f"Đã đồng bộ hóa dữ liệu. Tổng cộng {len(merged_df)} điểm dữ liệu.")
    
    # Chuẩn bị dữ liệu cuối cùng để phân tích
    merged_df['date'] = pd.to_datetime(merged_df['date'])  # Chuyển về datetime để vẽ biểu đồ
    merged_df = merged_df.sort_values('date')
    
    dates = merged_df['date']
    views = merged_df['view_main']  # Lấy cột view từ file nguồn
    y_true = merged_df['label'].to_numpy()
    
    # --- 3. LỰA CHỌN MÔ HÌNH TỐT NHẤT (KHÔNG GIÁM SÁT) ---
    print("\n--- Bắt đầu lựa chọn mô hình không giám sát ---")
    selected_model_name = main.select_best_model_unsupervised(views)
    # Cho phép override nhanh qua biến môi trường để test mô hình mới
    override = os.environ.get('ANOMALY_MODEL_OVERRIDE')
    if override:
        print(f"Override mô hình qua ENV: {override}")
        selected_model_name = override
    print(f"\n==> Mô hình được chọn: {selected_model_name}")

    # --- 4. CHẠY MÔ HÌNH ĐÃ CHỌN VÀ TÍNH F1-SCORE ---
    print("\n--- Chạy mô hình đã chọn để tạo dự đoán ---")
    
    # Tạo DataFrame cho input
    X_input = pd.DataFrame({'value': views})
    
    # Chạy dự đoán dựa trên mô hình được chọn
    if selected_model_name == 'SR':
        scores = anomaly_detection_base_model.run_sr_scores(X_input)
    elif selected_model_name == 'SH-ESD':
        scores = anomaly_detection_base_model.run_shesd_scores(X_input)
    elif selected_model_name == 'MA':
        scores = anomaly_detection_base_model.run_moving_average_scores(X_input, window_size=7)
    elif selected_model_name == 'IQR':
        scores = anomaly_detection_base_model.run_iqr_combined_scores(X_input)
    else:
        print(f"Mô hình {selected_model_name} không được hỗ trợ")
        exit()
    
    # Chuyển scores thành predictions
    threshold = np.percentile(scores, 100 * (1 - CONTAMINATION_RATE_ESTIMATE))
    y_pred = (scores > threshold).astype(int)
    
    # Tính F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"\n==> F1-Score của mô hình {selected_model_name} là: {f1:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Số anomaly thật: {y_true.sum()}")
    print(f"Số anomaly dự đoán: {y_pred.sum()}")

    # --- 5. TRỰC QUAN HÓA KẾT QUẢ ---
    print("\n--- Đang tạo biểu đồ trực quan hóa... ---")
    
    true_anomaly_dates = dates[y_true == 1]
    true_anomaly_views = views[y_true == 1]
    
    predicted_anomaly_dates = dates[y_pred == 1]
    predicted_anomaly_views = views[y_pred == 1]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))

    # Vẽ chuỗi thời gian chính
    ax.plot(dates, views, color='cornflowerblue', label='Lượt xem (Views)', zorder=1)
    
    # Đánh dấu các điểm bất thường thật
    ax.scatter(true_anomaly_dates, true_anomaly_views, 
               s=100, color='red', marker='o', 
               edgecolor='black', label='Bất thường thật', zorder=2)
               
    # Đánh dấu các điểm bất thường được dự đoán
    ax.scatter(predicted_anomaly_dates, predicted_anomaly_views,
               s=150, color='none', marker='o',
               edgecolor='limegreen', linewidth=2, label=f'Dự đoán ({selected_model_name})', zorder=3)
    
    # Cấu hình biểu đồ
    title = (f"Phát hiện bất thường cho PlaceID: {TEST_PLACE_ID}\n"
             f"Mô hình được chọn (Không giám sát): {selected_model_name} | F1-Score: {f1:.4f}")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Ngày", fontsize=12)
    ax.set_ylabel("Lượt xem", fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    # Lưu biểu đồ ra file
    output_filename = f"visualization_{TEST_PLACE_ID}.png"
    plt.savefig(output_filename)
    print(f"Biểu đồ đã được lưu vào file: {output_filename}")
    
    plt.show()
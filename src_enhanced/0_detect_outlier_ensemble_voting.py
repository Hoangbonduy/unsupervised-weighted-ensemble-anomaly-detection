import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

# --- YÊU CẦU QUAN TRỌNG ---
# Đảm bảo file `anomaly_detection_base_model.py` nằm cùng thư mục với file này.
try:
    import anomaly_detection_base_model as ad_models
except ImportError:
    print("LỖI: Không tìm thấy file 'anomaly_detection_base_model.py'.")
    print("Vui lòng đặt file đó vào cùng thư mục và chạy lại.")
    exit()

def detect_and_plot_ensemble_outliers(
    file_path,
    output_folder='ensemble_outlier_plots',
    vote_threshold=3,
    percentile_threshold=99.0
):
    """
    Phát hiện ngoại lai bằng phương pháp Ensemble Voting và vẽ biểu đồ.

    Args:
        file_path (str): Đường dẫn đến file CSV.
        output_folder (str): Thư mục lưu biểu đồ.
        vote_threshold (int): Số phiếu tối thiểu để một điểm bị coi là ngoại lai.
        percentile_threshold (float): Ngưỡng phân vị để xác định điểm bất thường từ scores.
    """
    # --- 1. Tải và chuẩn bị dữ liệu ---
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    df = df.sort_values(by=['placeId', 'date'])

    # --- 2. Định nghĩa "Hội đồng Bỏ phiếu" ---
    # Bao gồm các mô hình bạn muốn sử dụng.
    BASE_MODELS = {
        'SR': ad_models.run_sr_scores,
        'MA': ad_models.run_moving_average_scores,
        'IForest': ad_models.run_iforest_scores,
        'KNN': ad_models.run_knn_scores,
        'Prophet': ad_models.run_prophet_scores,
        'Moment': ad_models.run_moment_scores
    }
    print(f"Sử dụng {len(BASE_MODELS)} mô hình trong hội đồng bỏ phiếu.")

    # --- 3. Tạo thư mục output ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục: '{output_folder}'")

    # --- 4. Lặp qua từng địa điểm ---
    place_ids = df['placeId'].unique()[:10]  # Chỉ lấy 10 địa điểm đầu tiên
    print(f"\nBắt đầu xử lý {len(place_ids)} địa điểm đầu tiên với ngưỡng bỏ phiếu là {vote_threshold}...")

    for place_id in tqdm(place_ids, desc="Processing Places"):
        place_df = df[df['placeId'] == place_id].copy()
        if len(place_df) < 20: # Bỏ qua nếu chuỗi quá ngắn
            continue

        # Chuẩn bị dữ liệu đầu vào cho các mô hình
        series_df = pd.DataFrame({'value': place_df['view'].values})
        vote_counts = np.zeros(len(series_df))

        # --- 5. Bỏ phiếu ---
        for model_name, model_func in BASE_MODELS.items():
            try:
                # Lấy điểm anomaly scores từ mỗi mô hình
                scores = model_func(series_df)
                scores = np.nan_to_num(scores, nan=0) # Xử lý giá trị NaN nếu có

                if len(scores) > 0:
                    # Xác định ngưỡng để coi là bất thường (ví dụ: top 1%)
                    threshold = np.percentile(scores, percentile_threshold)
                    # Những điểm có score > ngưỡng sẽ được 1 phiếu
                    anomalies_indices = np.where(scores > threshold)[0]
                    vote_counts[anomalies_indices] += 1
            except Exception as e:
                # print(f"  - Cảnh báo: Mô hình {model_name} lỗi trên PlaceID {place_id}. Lỗi: {e}")
                continue

        # --- 6. Đếm phiếu và xác định ngoại lai cuối cùng ---
        final_outlier_indices = np.where(vote_counts >= vote_threshold)[0]
        outliers_df = place_df.iloc[final_outlier_indices]

        # --- 7. Vẽ và lưu biểu đồ ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 6))

        ax.plot(place_df['date'], place_df['view'], label='Chuỗi thời gian gốc', color='royalblue', zorder=1)
        ax.scatter(outliers_df['date'], outliers_df['view'], color='red', s=60, label=f'Điểm ngoại lai (>= {vote_threshold} phiếu)', zorder=2)

        ax.set_title(f"Phát hiện ngoại lai cho PlaceID: {place_id} bằng Ensemble Voting", fontsize=14)
        ax.set_xlabel("Ngày", fontsize=12)
        ax.set_ylabel("Lượt xem (view)", fontsize=12)
        ax.legend()
        plt.tight_layout()

        output_path = os.path.join(output_folder, f"ensemble_outliers_{place_id}.png")
        plt.savefig(output_path)
        plt.close(fig)

    print(f"\nHoàn thành! Tất cả biểu đồ đã được lưu vào thư mục '{output_folder}'.")

# --- HƯỚNG DẪN SỬ DỤNG ---
if __name__ == '__main__':
    # 1. Thay 'your_data.csv' bằng tên file của bạn.
    csv_file_name = 'data/cleaned_data_no_zero_periods_filtered.csv'
    
    # 2. (Tùy chọn) Điều chỉnh ngưỡng bỏ phiếu. 
    #    - Ngưỡng cao hơn (4, 5): Chỉ bắt những điểm bất thường rõ ràng nhất.
    #    - Ngưỡng thấp hơn (2, 3): Bắt nhiều điểm hơn, có thể nhạy hơn.
    VOTE_THRESHOLD = 3 

    detect_and_plot_ensemble_outliers(
        file_path=csv_file_name, 
        vote_threshold=VOTE_THRESHOLD
    )
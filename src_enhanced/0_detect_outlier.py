import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy import interpolate

# --- YÊU CẦU QUAN TRỌNG ---
# Đảm bảo file `anomaly_detection_base_model.py` nằm cùng thư mục với file này.
try:
    import anomaly_detection_base_model as ad_models
except ImportError:
    print("LỖI: Không tìm thấy file 'anomaly_detection_base_model.py'.")
    print("Vui lòng đặt file đó vào cùng thư mục và chạy lại.")
    exit()

def interpolate_outliers(series, outlier_indices):
    """
    Nội suy các điểm bất thường trong chuỗi thời gian.
    
    Args:
        series (pd.Series): Chuỗi thời gian gốc
        outlier_indices (np.array): Chỉ số các điểm bất thường
    
    Returns:
        pd.Series: Chuỗi thời gian đã được làm sạch
    """
    if len(outlier_indices) == 0:
        return series.copy()
    
    clean_series = series.copy()
    
    # Tạo mask cho các điểm bình thường
    normal_mask = np.ones(len(series), dtype=bool)
    normal_mask[outlier_indices] = False
    
    # Lấy các điểm bình thường để nội suy
    normal_indices = np.where(normal_mask)[0]
    normal_values = series.iloc[normal_indices]
    
    if len(normal_indices) < 2:
        # Nếu quá ít điểm bình thường, không thể nội suy
        return series.copy()
    
    # Thực hiện nội suy tuyến tính
    f = interpolate.interp1d(normal_indices, normal_values, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    # Thay thế các điểm bất thường bằng giá trị nội suy
    clean_series.iloc[outlier_indices] = f(outlier_indices)
    
    return clean_series

def detect_iqr_outliers(series):
    """
    Phát hiện điểm bất thường bằng phương pháp IQR.
    
    Args:
        series (pd.Series): Chuỗi thời gian
    
    Returns:
        np.array: Chỉ số các điểm bất thường
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return np.where(outlier_mask)[0]

def detect_ensemble_outliers(series, vote_threshold=3, percentile_threshold=99.0):
    """
    Phát hiện điểm bất thường bằng phương pháp Ensemble Voting.
    
    Args:
        series (pd.Series): Chuỗi thời gian
        vote_threshold (int): Số phiếu tối thiểu để coi là bất thường
        percentile_threshold (float): Ngưỡng phân vị
    
    Returns:
        np.array: Chỉ số các điểm bất thường
    """
    # Định nghĩa các mô hình
    BASE_MODELS = {
        'SR': ad_models.run_sr_scores,
        'MA': ad_models.run_moving_average_scores,
        'IForest': ad_models.run_iforest_scores,
        'KNN': ad_models.run_knn_scores,
        'Prophet': ad_models.run_prophet_scores,
        'Moment': ad_models.run_moment_scores
    }
    
    # Chuẩn bị dữ liệu đầu vào cho các mô hình
    series_df = pd.DataFrame({'value': series.values})
    vote_counts = np.zeros(len(series))
    
    # Bỏ phiếu
    for model_name, model_func in BASE_MODELS.items():
        try:
            # Lấy điểm anomaly scores từ mỗi mô hình
            scores = model_func(series_df)
            scores = np.nan_to_num(scores, nan=0)
            
            if len(scores) > 0:
                # Xác định ngưỡng để coi là bất thường
                threshold = np.percentile(scores, percentile_threshold)
                # Những điểm có score > ngưỡng sẽ được 1 phiếu
                anomalies_indices = np.where(scores > threshold)[0]
                vote_counts[anomalies_indices] += 1
        except Exception as e:
            continue
    
    # Xác định điểm bất thường cuối cùng
    final_outlier_indices = np.where(vote_counts >= vote_threshold)[0]
    return final_outlier_indices

def detect_and_clean_outliers(
    file_path,
    output_folder='outlier_detection_plots',
    vote_threshold=3,
    percentile_threshold=99.0
):
    """
    Phát hiện và làm sạch điểm bất thường theo quy trình 2 bước:
    1. IQR detection + interpolation
    2. Ensemble Voting detection + interpolation
    
    Args:
        file_path (str): Đường dẫn đến file CSV
        output_folder (str): Thư mục lưu biểu đồ
        vote_threshold (int): Số phiếu tối thiểu cho ensemble voting
        percentile_threshold (float): Ngưỡng phân vị cho ensemble voting
    """
    # --- 1. Tải và chuẩn bị dữ liệu ---
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file tại '{file_path}'.")
        return
    
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
    except Exception as e:
        print(f"LỖI: Không thể đọc file CSV. Lỗi: {e}")
        return
    
    df = df.sort_values(by=['placeId', 'date'])
    
    # --- 2. Tạo thư mục output ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục: '{output_folder}'")
    
    # --- 3. Xử lý tất cả các địa điểm ---
    place_ids = df['placeId'].unique()
    print(f"\nBắt đầu xử lý {len(place_ids)} địa điểm...")
    print("Quy trình: IQR detection → Interpolation → Ensemble Voting → Interpolation")
    
    results = []
    cleaned_data = []  # Lưu dữ liệu đã làm sạch
    
    for i, place_id in enumerate(tqdm(place_ids, desc="Processing Places")):
        place_df = df[df['placeId'] == place_id].copy()
        
        if len(place_df) < 20:  # Bỏ qua nếu chuỗi quá ngắn
            continue
        
        # Dữ liệu gốc
        original_series = place_df['view'].copy()
        dates = place_df['date'].copy()
        
        # --- BƯỚC 1: IQR Detection + Interpolation ---
        iqr_outliers = detect_iqr_outliers(original_series)
        series_after_iqr = interpolate_outliers(original_series, iqr_outliers)
        
        # --- BƯỚC 2: Ensemble Voting Detection + Interpolation ---
        ensemble_outliers = detect_ensemble_outliers(series_after_iqr, vote_threshold, percentile_threshold)
        final_clean_series = interpolate_outliers(series_after_iqr, ensemble_outliers)
        
        # --- Lưu dữ liệu đã làm sạch ---
        for j, (date, clean_value) in enumerate(zip(dates, final_clean_series)):
            cleaned_data.append({
                'placeId': place_id,
                'date': date,
                'view': int(round(clean_value))  # Làm tròn thành số nguyên
            })
        
        # --- 4. Lưu kết quả cho visualization (chỉ 10 địa điểm đầu) ---
        if i < 10:  # Chỉ vẽ biểu đồ cho 10 địa điểm đầu tiên
            results.append({
                'place_id': place_id,
                'original_series': original_series,
                'after_iqr': series_after_iqr,
                'final_clean': final_clean_series,
                'dates': dates,
                'iqr_outliers': iqr_outliers,
                'ensemble_outliers': ensemble_outliers
            })
            
            # --- 5. Vẽ và lưu biểu đồ ---
            # --- 5. Vẽ và lưu biểu đồ ---
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            
            # Biểu đồ 1: Dữ liệu gốc + IQR outliers
            axes[0].plot(dates, original_series, label='Dữ liệu gốc', color='blue', alpha=0.7)
            if len(iqr_outliers) > 0:
                axes[0].scatter(dates.iloc[iqr_outliers], original_series.iloc[iqr_outliers], 
                              color='red', s=60, label=f'IQR Outliers ({len(iqr_outliers)})', zorder=3)
            axes[0].set_title(f"PlaceID {place_id} - Bước 1: IQR Detection", fontsize=12)
            axes[0].set_ylabel("Lượt xem", fontsize=10)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Biểu đồ 2: Sau IQR cleaning + Ensemble outliers
            axes[1].plot(dates, series_after_iqr, label='Sau IQR cleaning', color='green', alpha=0.7)
            if len(ensemble_outliers) > 0:
                axes[1].scatter(dates.iloc[ensemble_outliers], series_after_iqr.iloc[ensemble_outliers], 
                              color='orange', s=60, label=f'Ensemble Outliers ({len(ensemble_outliers)})', zorder=3)
            axes[1].set_title(f"Bước 2: Ensemble Voting Detection (threshold={vote_threshold})", fontsize=12)
            axes[1].set_ylabel("Lượt xem", fontsize=10)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Biểu đồ 3: So sánh cuối cùng
            axes[2].plot(dates, original_series, label='Dữ liệu gốc', color='blue', alpha=0.5)
            axes[2].plot(dates, final_clean_series, label='Dữ liệu đã làm sạch', color='darkgreen', linewidth=2)
            axes[2].set_title("Bước 3: Kết quả cuối cùng", fontsize=12)
            axes[2].set_xlabel("Ngày", fontsize=10)
            axes[2].set_ylabel("Lượt xem", fontsize=10)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(output_folder, f"outlier_cleaning_{place_id}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # --- Thống kê ---
        if i < 10:  # Chỉ in thống kê cho 10 địa điểm đầu tiên
            print(f"\nPlaceID {place_id}:")
            print(f"  - IQR outliers: {len(iqr_outliers)}")
            print(f"  - Ensemble outliers: {len(ensemble_outliers)}")
            print(f"  - Total cleaned points: {len(iqr_outliers) + len(ensemble_outliers)}")
    
    # --- 6. Lưu dữ liệu đã làm sạch ra file CSV ---
    cleaned_df = pd.DataFrame(cleaned_data)
    output_csv_path = 'data/cleaned_data_smoothed.csv'
    cleaned_df.to_csv(output_csv_path, index=False)
    print(f"\nĐã lưu dữ liệu đã làm sạch vào: {output_csv_path}")
    print(f"Tổng số dòng dữ liệu: {len(cleaned_df)}")
    
    print(f"\nHoàn thành! Tất cả biểu đồ đã được lưu vào thư mục '{output_folder}'.")
    print(f"Đã xử lý {len(place_ids)} địa điểm (visualization cho 10 địa điểm đầu tiên).")
    
    return results, cleaned_df

if __name__ == '__main__':
    # Cấu hình
    csv_file_name = 'data/cleaned_data_after_idx30.csv'
    VOTE_THRESHOLD = 3  # Ngưỡng bỏ phiếu cho ensemble
    PERCENTILE_THRESHOLD = 98.0  # Ngưỡng phân vị
    
    # Chạy quy trình phát hiện và làm sạch
    results, cleaned_df = detect_and_clean_outliers(
        file_path=csv_file_name,
        vote_threshold=VOTE_THRESHOLD,
        percentile_threshold=PERCENTILE_THRESHOLD
    )
    
    print(f"\nTóm tắt:")
    print(f"- Số địa điểm đã xử lý: {len(cleaned_df['placeId'].unique()) if len(cleaned_df) > 0 else 0}")
    print(f"- Tổng số điểm dữ liệu: {len(cleaned_df) if len(cleaned_df) > 0 else 0}")
    print(f"- File đầu ra: data/cleaned_data_smoothed.csv")
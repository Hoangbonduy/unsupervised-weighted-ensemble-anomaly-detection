import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

# --- PHẦN 0: IMPORT CÁC THƯ VIỆN VÀ MÔ HÌNH ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import anomaly_detection_base_model as ad_models
except ImportError:
    print("LỖI: Không tìm thấy file 'anomaly_detection_base_model.py'.")
    exit()

# --- PHẦN 1: CÁC HÀM TIỆN ÍCH ---

def normalize_scores(scores):
    """Chuẩn hóa điểm số về khoảng [0, 1] để kết hợp."""
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val - min_val > 1e-6:
        return (scores - min_val) / (max_val - min_val)
    return np.zeros_like(scores)

def compute_dynamic_threshold(scores: np.ndarray, contamination: float = 0.05) -> float:
    """Xác định ngưỡng dựa trên phân vị."""
    if scores.size == 0: return np.inf
    # Sử dụng phân vị để xác định ngưỡng, ổn định hơn z-score
    return np.percentile(scores, 100 * (1 - contamination))

def visualize_ensemble_result(series, true_labels, predicted_labels, place_id):
    """Vẽ đồ thị so sánh kết quả ensemble với nhãn đúng."""
    plt.figure(figsize=(20, 8))
    
    # Vẽ chuỗi thời gian gốc
    plt.plot(series, label='Original Time Series', color='royalblue', alpha=0.8, zorder=2)
    
    true_indices = np.where(true_labels == 1)[0]
    predicted_indices = np.where(predicted_labels == 1)[0]
    
    # Đánh dấu các vùng bất thường thực sự (Ground Truth)
    if len(true_indices) > 0:
        for i, idx in enumerate(true_indices):
            # Dùng vspan để tô màu cả một vùng
            plt.axvspan(idx - 0.5, idx + 0.5, color='gold', alpha=0.6, zorder=1, 
                        label='Ground Truth Anomaly' if i == 0 else "")

    # Đánh dấu các điểm bất thường do mô hình dự đoán
    if len(predicted_indices) > 0:
        plt.scatter(predicted_indices, series[predicted_indices], 
                    color='red', marker='o', s=60, zorder=3, 
                    label='Ensemble Prediction')

    plt.title(f'Weighted Ensemble Anomaly Detection vs. Ground Truth for PlaceID: {place_id}', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('View', fontsize=12)
    
    # Xử lý legend để không bị lặp
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_file = f'ensemble_result_placeid_{place_id}.png'
    plt.savefig(output_file)
    print(f"\nĐã lưu đồ thị kết quả vào file: {output_file}")
    plt.close()

# --- PHẦN 2: LOGIC CHÍNH ---

DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
LABEL_DIR = 'labels'
TARGET_PLACEID = 4611864748268400448

def main():
    # 1. KẾT QUẢ ĐÁNH GIÁ VUS-ROC TỪ BƯỚC TRƯỚC
    # Dữ liệu này được dùng để tính trọng số
    vus_results = {
        'SR':       0.631, 'IQR':      0.555, 'MA':       0.582,
        'IForest':  0.654, 'KNN':      0.658, 'RePAD':    0.618,
        'Prophet':  0.544, 'Moment':   0.699
    }
    
    # 2. TÍNH TRỌNG SỐ CHO MỖI MÔ HÌNH
    # Trọng số là điểm VUS-ROC trung bình, được chuẩn hóa để tổng bằng 1
    total_vus_score = sum(vus_results.values())
    weights = {model: score / total_vus_score for model, score in vus_results.items()}
    
    print("="*50)
    print("TRỌNG SỐ CỦA CÁC MÔ HÌNH (dựa trên VUS-ROC)")
    print("="*50)
    for model, weight in weights.items():
        print(f"- {model:<10}: {weight:.3f}")
    print("="*50)

    # 3. TẢI DỮ LIỆU VÀ NHÃN THỰC TẾ
    data_full = pd.read_csv(DATA_FILE_PATH)
    ts_group = data_full[data_full['placeId'] == TARGET_PLACEID].sort_values('date')
    original_series = ts_group['view'].to_numpy().astype(float)
    
    label_path = os.path.join(LABEL_DIR, f'label_{TARGET_PLACEID}.csv')
    try:
        label_df = pd.read_csv(label_path)
        # Đảm bảo nhãn và dữ liệu có cùng độ dài
        true_labels = label_df['label'].to_numpy()[:len(original_series)]
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file nhãn tại '{label_path}'.")
        return
        
    print(f"\nĐã tải dữ liệu và nhãn cho PlaceID: {TARGET_PLACEID}. Độ dài: {len(original_series)} điểm.")
    print(f"Số điểm bất thường thực tế: {np.sum(true_labels)}")

    # 4. CHẠY TẤT CẢ MÔ HÌNH VÀ KẾT HỢP KẾT QUẢ
    models_to_run = {
        'SR': ad_models.run_sr_scores, 'IQR': ad_models.run_iqr_scores, 'MA': ad_models.run_moving_average_scores,
        'IForest': ad_models.run_iforest_scores, 'KNN': ad_models.run_knn_scores, 'RePAD': ad_models.run_repad_scores,
        'Prophet': ad_models.run_prophet_scores, 'Moment': ad_models.run_moment_scores
    }
    
    ensemble_scores = np.zeros_like(original_series, dtype=float)
    series_df = pd.DataFrame(original_series)

    for model_name, model_func in tqdm(models_to_run.items(), desc="Chạy các mô hình và kết hợp"):
        try:
            # Lấy điểm bất thường từ mô hình
            scores = model_func(series_df)
            # Chuẩn hóa điểm số
            normalized = normalize_scores(np.array(scores))
            # Cộng vào điểm ensemble với trọng số tương ứng
            ensemble_scores += normalized * weights[model_name]
        except Exception as e:
            print(f"\nLỗi khi chạy mô hình {model_name}: {e}")
            continue
            
    # 5. ĐƯA RA DỰ ĐOÁN CUỐI CÙNG
    # Xác định ngưỡng cho điểm ensemble
    threshold = compute_dynamic_threshold(ensemble_scores)
    predicted_labels = (ensemble_scores > threshold).astype(int)
    
    print(f"\nĐã hoàn thành Weighted Ensemble.")
    print(f"Ngưỡng dự đoán được xác định: {threshold:.4f}")
    print(f"Tổng số điểm bất thường dự đoán: {np.sum(predicted_labels)}")

    # 6. TRỰC QUAN HÓA KẾT QUẢ
    visualize_ensemble_result(original_series, true_labels, predicted_labels, TARGET_PLACEID)

if __name__ == '__main__':
    main()
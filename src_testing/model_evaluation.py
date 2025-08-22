import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
import torch

# --- PHẦN 0: IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import trực tiếp từ thư viện timeeval đã cài đặt
try:
    from timeeval.metrics.vus_metrics import RangeRocVUS
except ImportError:
    print("LỖI: Không tìm thấy thư viện 'timeeval'.")
    print("Vui lòng cài đặt bằng lệnh: pip install timeeval")
    exit()

# Import các hàm mô hình từ file của bạn
try:
    import anomaly_detection_base_model as ad_models
except ImportError:
    print("LỖI: Không tìm thấy file 'anomaly_detection_base_model.py'.")
    print("Vui lòng đảm bảo file này nằm cùng thư mục.")
    exit()

# --- PHẦN 1: CÁC HÀM THÊM BẤT THƯỜNG (4 LOẠI CỐT LÕI) ---

class AnomalyInjector:
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(random_state)

    def _get_anomaly_location(self, series, length):
        # Đảm bảo series đủ dài để thực hiện rolling window
        window_size = max(10, length * 2)
        if len(series) < window_size:
             if len(series) > length:
                return self.rng.integers(0, len(series) - length)
             else:
                return 0
        moving_std = pd.Series(series).rolling(window=window_size, center=True).std().bfill().ffill().to_numpy()
        candidates = np.where(moving_std < np.quantile(moving_std, 0.25))[0]
        if len(candidates) > length and len(series) > length:
            start_idx = self.rng.choice(candidates)
            return min(start_idx, len(series) - length)
        elif len(series) > length:
            return self.rng.integers(0, len(series) - length)
        else:
            return 0


    def inject(self, series, anomaly_type, params):
        # Bỏ 'noise' ra khỏi danh sách
        method_map = {
            "global": self._inject_global, "contextual": self._inject_contextual,
            "trend": self._inject_trend, "seasonal": self._inject_seasonal,
            "cutoff": self._inject_cutoff, "average": self._inject_average
        }
        return method_map[anomaly_type](series, params)

    # --- Các phương pháp được giữ lại ---
    def _inject_global(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        indices = self.rng.choice(len(series), params["n_anomalies"], replace=False)
        for idx in indices:
            series[idx] += self.rng.choice([-1, 1]) * params["magnitude_std"] * np.std(series)
            labels[idx] = 1
        return series, labels

    def _inject_contextual(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        indices = self.rng.choice(len(series), params["n_anomalies"], replace=False)
        for idx in indices:
            series[idx] = np.mean(series) + self.rng.choice([-1, 1]) * params["magnitude_std"] * np.std(series)
            labels[idx] = 1
        return series, labels

    def _inject_trend(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        trend = np.linspace(0, self.rng.choice([-1, 1]) * params["magnitude"], length)
        series[start:start+length] += trend
        labels[start:start+length] = 1
        return series, labels

    def _inject_seasonal(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        period = params["period"]
        if len(series) < period * 2: return series, labels
        try:
            seasonal = seasonal_decompose(series, model='additive', period=period).seasonal
            peaks, _ = find_peaks(seasonal, height=0)
            if len(peaks) == 0: return series, labels
            indices = self.rng.choice(peaks, min(params["n_anomalies"], len(peaks)), replace=False)
            for idx in indices:
                series[idx] += self.rng.choice([-1, 1]) * params["magnitude_std"] * np.std(series)
                labels[idx] = 1
        except Exception:
            return series, labels
        return series, labels

    def _inject_cutoff(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        cutoff_value = np.quantile(series, 0.05)
        series[start:start+length] = cutoff_value
        labels[start:start+length] = 1
        return series, labels

    def _inject_average(self, series, params):
        series, labels = series.copy(), np.zeros_like(series, dtype=int)
        length = self.rng.integers(params["min_len"], params["max_len"])
        start = self._get_anomaly_location(series, length)
        window = params.get("smoothing_window", 5)
        segment = series[start:start+length]
        smoothed_segment = pd.Series(segment).rolling(window, center=True, min_periods=1).mean().to_numpy()
        series[start:start+length] = smoothed_segment
        labels[start:start+length] = 1
        return series, labels

# --- PHẦN 2: LOGIC ĐÁNH GIÁ MÔ HÌNH ---

DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
TARGET_PLACEID = 4611864748268400448

def main():
    print(f"Bắt đầu đánh giá mô hình trên PlaceID: {TARGET_PLACEID} (sử dụng thư viện 'timeeval')...")
    
    data_full = pd.read_csv(DATA_FILE_PATH)
    original_series = data_full[data_full['placeId'] == TARGET_PLACEID].sort_values('date')['view'].to_numpy().astype(float)
    n = len(original_series)
    print(f"Đã tải dữ liệu, độ dài: {n} điểm.")

    injector = AnomalyInjector(random_state=42)
    
    models_to_evaluate = {
        'SR': ad_models.run_sr_scores, 'IQR': ad_models.run_iqr_scores, 'MA': ad_models.run_moving_average_scores,
        'IForest': ad_models.run_iforest_scores, 'KNN': ad_models.run_knn_scores, 'RePAD': ad_models.run_repad_scores,
        'Prophet': ad_models.run_prophet_scores, 'Moment': ad_models.run_moment_scores
    }

    obvious_profile = {
        "global": {"n_anomalies": 3, "magnitude_std": 7.0},
        "contextual": {"n_anomalies": 3, "magnitude_std": 5.0},
        "trend": {"min_len": n//15, "max_len": n//8, 
        "magnitude": np.std(original_series) * 2.0},
        "seasonal": {"n_anomalies": 4, "magnitude_std": 6.0, "period": 7},
        "cutoff": {"min_len": n//20, "max_len": n//10},
        "average": {"min_len": n//15, "max_len": n//8}
    }
    results = {}
    
    # Khởi tạo đối tượng tính toán VUS-ROC từ timeeval
    # max_buffer_size là tham số tương đương với max_ell trong các phiên bản trước
    vus_roc_calculator = RangeRocVUS(max_buffer_size=n // 10, compatibility_mode=True)
    
    for anomaly_type, params in tqdm(obvious_profile.items(), desc="Đánh giá các loại bất thường"):
        injected_series, pseudo_labels = injector.inject(original_series, anomaly_type, params)
        injected_df = pd.DataFrame(injected_series)
        
        vus_scores = {}
        for model_name, model_func in models_to_evaluate.items():
            try:
                anomaly_scores = model_func(injected_df)
                
                # Chuyển đổi sang kiểu dữ liệu timeeval yêu cầu
                y_true = pseudo_labels.astype(np.float64)
                y_score = np.array(anomaly_scores).astype(np.float64)
                
                # SỬ DỤNG VUS-ROC TỪ TIMEVAL
                if np.sum(y_true) > 0:
                    score = vus_roc_calculator(y_true=y_true, y_score=y_score)
                else:
                    score = 0.0 # Không có bất thường để đánh giá
                
                vus_scores[model_name] = score
            except Exception as e:
                print(f"Lỗi khi chạy {model_name} trên {anomaly_type}: {e}")
                vus_scores[model_name] = np.nan
        
        results[anomaly_type] = vus_scores

    results_df = pd.DataFrame(results).T
    results_df['Average'] = results_df.mean(axis=1)
    results_df.loc['Average'] = results_df.mean(axis=0)

    print("\n" + "="*80)
    print("KẾT QUẢ ĐÁNH GIÁ VUS-ROC (sử dụng thư viện 'timeeval')")
    print("="*80)
    print(results_df.round(3).to_string())
    print("="*80)
    
    output_file = f'timeeval_vus_results_placeid_{TARGET_PLACEID}.csv'
    results_df.to_csv(output_file)
    print(f"\nĐã lưu kết quả chi tiết vào file: {output_file}")

if __name__ == '__main__':
    main()
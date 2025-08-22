import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score

def find_dominant_period(series, min_period=4, max_period=100):
    """Sử dụng Fast Fourier Transform (FFT) để tìm chu kỳ trội trong chuỗi."""
    if len(series) < 3: return 0
    fft_series = np.fft.fft(series)[1:]
    frequencies = np.fft.fftfreq(len(series))[1:]
    if len(fft_series) == 0: return 0
    
    peak_coeff = np.argmax(np.abs(fft_series))
    peak_freq = frequencies[peak_coeff]
    period = 1 / abs(peak_freq) if peak_freq != 0 else 0
    
    return int(np.clip(period, min_period, max_period))

def get_injection_location(series_len, anomaly_len, period):
    """Chọn một vị trí tiêm thông minh: ưu tiên đầu các chu kỳ."""
    if period == 0 or series_len < period * 2 or anomaly_len >= series_len:
        return np.random.randint(0, max(1, series_len - anomaly_len))
    
    valid_cycle_starts = np.arange(0, series_len - anomaly_len, period)
    if len(valid_cycle_starts) == 0:
        return np.random.randint(0, max(1, series_len - anomaly_len))
    
    return np.random.choice(valid_cycle_starts)

# --- Các hàm inject_* ---

def inject_spike_anomaly(time_series):
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    if len(ts_copy) < 5: return ts_copy, pseudo_labels
    
    num_spikes = np.random.randint(1, 4)
    strength = np.random.uniform(2.5, 4.0)
    
    injection_points = np.random.choice(len(ts_copy), size=num_spikes, replace=False)
    spike_magnitude = np.std(ts_copy) * strength
    for point in injection_points:
        ts_copy[point] += spike_magnitude * np.random.choice([-1, 1])
        pseudo_labels[point] = 1
    return ts_copy, pseudo_labels

def inject_level_shift_anomaly(time_series):
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.05, 0.15))
    strength = np.random.uniform(1.5, 2.5)
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels

    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len

    shift_magnitude = np.std(ts_copy) * strength * np.random.choice([-1, 1])
    ts_copy[start:end] += shift_magnitude
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_dip_anomaly(time_series):
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)

    anomaly_len = int(series_len * np.random.uniform(0.05, 0.15))
    strength = np.random.uniform(2.0, 3.5)

    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels

    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len

    dip_magnitude = np.std(ts_copy) * strength
    ts_copy[start:end] -= dip_magnitude # Luôn trừ đi
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_flip_anomaly(time_series):
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)

    anomaly_len = int(series_len * np.random.uniform(0.05, 0.1))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels

    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len

    ts_copy[start:end] = ts_copy[start:end][::-1]
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_noise_anomaly(time_series):
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    strength = np.random.uniform(0.3, 0.6)
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    local_std = np.std(ts_copy[start:end])
    noise = np.random.normal(loc=0, scale=local_std * strength, size=anomaly_len)
    ts_copy[start:end] += noise
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

def inject_scale_anomaly(time_series):
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    scale_factor = np.random.choice([np.random.uniform(1.5, 2.5), np.random.uniform(0.2, 0.5)])
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    ts_copy[start:end] *= scale_factor
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

# MỚI: Contextual Anomaly - Thay đổi cục bộ không phù hợp với ngữ cảnh
def inject_contextual_anomaly(time_series):
    """Áp dụng một biến đổi tuyến tính nhỏ (a*Y + b) vào một đoạn dữ liệu[cite: 2118]."""
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)

    anomaly_len = int(series_len * np.random.uniform(0.05, 0.15))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    # a ~ N(1, 0.1^2), b ~ N(0, (0.1*std)^2) [cite: 2118]
    a = np.random.normal(1, 0.1)
    b = np.random.normal(0, 0.1 * np.std(ts_copy))
    
    ts_copy[start:end] = a * ts_copy[start:end] + b
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

# MỚI: Speedup Anomaly - Tăng/giảm tần số của một đoạn
def inject_speedup_anomaly(time_series):
    """Tăng (hoặc giảm) tốc độ của một đoạn dữ liệu bằng cách nội suy[cite: 2089]."""
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    # Tăng tốc (factor > 1) hoặc giảm tốc (factor < 1)
    speed_factor = np.random.choice([np.random.uniform(1.5, 2.5), np.random.uniform(0.4, 0.7)])
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    segment = ts_copy[start:end]
    x = np.linspace(0, 1, len(segment))
    f = interp1d(x, segment, kind='linear', fill_value="extrapolate")
    
    # Tạo index mới đã được tăng/giảm tốc
    x_new = np.linspace(0, 1, int(len(segment) / speed_factor))
    segment_resampled = f(x_new)
    
    # Nội suy lại để vừa với độ dài ban đầu
    f_new = interp1d(np.linspace(0, 1, len(segment_resampled)), segment_resampled, kind='linear', fill_value="extrapolate")
    ts_copy[start:end] = f_new(x)
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

# MỚI: Cutoff Anomaly - Cắt và thay bằng giá trị gần hằng số
def inject_cutoff_anomaly(time_series):
    """Thay thế một đoạn dữ liệu bằng một giá trị gần như hằng số[cite: 2110]."""
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.05, 0.15))
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    # Thay bằng giá trị tại điểm bắt đầu + một chút nhiễu nhỏ
    cutoff_value = ts_copy[start] if start < len(ts_copy) else np.mean(ts_copy)
    noise = np.random.normal(0, 0.05 * np.std(ts_copy), anomaly_len)
    
    ts_copy[start:end] = cutoff_value + noise
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

# MỚI: Wander Anomaly - Thêm một xu hướng tuyến tính cục bộ
def inject_wander_anomaly(time_series):
    """Thêm một trend tuyến tính vào một đoạn dữ liệu[cite: 2117]."""
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    strength = np.random.uniform(1.0, 2.0)
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    # Baseline là mức độ thay đổi từ đầu đến cuối đoạn
    baseline = np.std(ts_copy) * strength * np.random.choice([-1, 1])
    trend = np.linspace(0, baseline, anomaly_len)
    
    ts_copy[start:end] += trend
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels
    
# MỚI: Average Anomaly - Làm mịn một đoạn, loại bỏ chi tiết
def inject_average_anomaly(time_series):
    """Thay thế một đoạn dữ liệu bằng trung bình trượt của chính nó [cite: 2111-2112]."""
    ts_copy = np.array(time_series, dtype=np.float64)
    pseudo_labels = np.zeros(len(ts_copy))
    series_len = len(ts_copy)
    
    anomaly_len = int(series_len * np.random.uniform(0.1, 0.2))
    window = max(3, int(anomaly_len * 0.2)) # Cửa sổ trượt là 20% độ dài đoạn
    
    if series_len < anomaly_len * 2: return ts_copy, pseudo_labels
    
    period = find_dominant_period(ts_copy)
    start = get_injection_location(series_len, anomaly_len, period)
    end = start + anomaly_len
    
    segment_series = pd.Series(ts_copy[start:end])
    moving_avg = segment_series.rolling(window=window, min_periods=1, center=True).mean().values
    
    ts_copy[start:end] = moving_avg
    pseudo_labels[start:end] = 1
    return ts_copy, pseudo_labels

# ==============================================================================
# PHẦN 2: HÀM CHÍNH get_synthetic_ranking
# ==============================================================================

def get_synthetic_ranking(time_series, models_to_run_scores):
    """
    Chạy các mô hình trên dữ liệu đã tiêm nhiều loại bất thường và xếp hạng chúng
    BẰNG CÁCH SỬ DỤNG AUC SCORE để đánh giá.
    """
    # Danh sách các bài kiểm tra (giữ nguyên)
    injection_tests = {
        'spike': inject_spike_anomaly,
        'level_shift': inject_level_shift_anomaly,
        'flip': inject_flip_anomaly,
        'noise': inject_noise_anomaly,
        'dip': inject_dip_anomaly
    }
    
    # Dictionary để lưu AUC-score của mỗi mô hình trên mỗi bài test
    all_auc_scores = {name: [] for name in models_to_run_scores.keys()}
    
    # --- Chạy qua từng bài kiểm tra ---
    for test_name, inject_func in injection_tests.items():
        # 1. Tạo dữ liệu kiểm tra: tiêm bất thường và lấy nhãn giả
        ts_injected, pseudo_labels = inject_func(time_series)
        ts_injected_df = pd.DataFrame(ts_injected)
        
        # Kiểm tra xem có cả lớp 0 và 1 trong nhãn giả không
        if len(np.unique(pseudo_labels)) < 2:
            # Nếu chỉ có 1 lớp, AUC không xác định. Gán điểm trung bình 0.5
            for model_name in models_to_run_scores.keys():
                all_auc_scores[model_name].append(0.5)
            continue

        # 2. Chạy tất cả các mô hình trên dữ liệu đã tiêm
        for model_name, run_func in models_to_run_scores.items():
            # Lấy điểm số bất thường thô từ mô hình
            anomaly_scores = run_func(ts_injected_df)
            
            # Xử lý giá trị infinity và NaN
            anomaly_scores = np.array(anomaly_scores)
            anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 3. Tính AUC score và lưu lại
            # roc_auc_score(y_true, y_scores)
            if len(anomaly_scores) == len(pseudo_labels) and np.all(np.isfinite(anomaly_scores)):
                try:
                    auc = roc_auc_score(pseudo_labels, anomaly_scores)
                    all_auc_scores[model_name].append(auc)
                except ValueError as e:
                    print(f"    Warning: AUC calculation failed for {model_name} on {test_name}: {e}")
                    all_auc_scores[model_name].append(0.0)
            else:
                print(f"    Warning: Invalid anomaly_scores for {model_name} on {test_name}")
                all_auc_scores[model_name].append(0.0) # Trường hợp lỗi

    # --- Tính toán điểm trung bình và xếp hạng ---
    avg_scores = {name: np.mean(auc_list) for name, auc_list in all_auc_scores.items()}
    ranked_models = sorted(avg_scores, key=avg_scores.get, reverse=True)
    
    # In ra bảng điểm chi tiết để debug
    print("\n  Điểm AUC chi tiết trên các bài test tổng hợp:")
    print(pd.DataFrame(all_auc_scores, index=injection_tests.keys()).round(3))
    print("\n  Điểm AUC trung bình:")
    for model_name in ranked_models:
        print(f"    - {model_name}: {avg_scores[model_name]:.4f}")

    return ranked_models, avg_scores
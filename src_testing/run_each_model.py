import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import t
import matplotlib.pyplot as plt
import warnings
import sranodec as anom
import joblib
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
import torch
from torch.utils.data import Dataset, DataLoader
from momentfm import MOMENTPipeline

# --- PHẦN 1: CÀI ĐẶT CHUNG VÀ CÁC HÀM MÔ HÌNH ---

# Tắt các cảnh báo không cần thiết để output gọn gàng hơn
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from prophet import Prophet

# --- Các hằng số và biến toàn cục ---
WEIGHTS_DIR = 'pretrained_weights/'
_REPAD_MODEL = None
_DEEPANT_MODEL = None
MOMENT_MODEL = None
DEVICE = None

# --- Các hàm tiện ích ---
def _to_numpy(x):
    if isinstance(x, np.ndarray): return x
    elif isinstance(x, pd.Series): return x.values
    elif isinstance(x, pd.DataFrame): return x.iloc[:, 0].values
    else: return np.array(x)

def statistical_fallback(series):
    series = _to_numpy(series)
    mean_val = np.mean(series)
    std_val = np.std(series)
    if std_val == 0: return np.zeros(len(series))
    return np.abs((series - mean_val) / std_val)

# --- Các hàm mô hình (được gộp từ anomaly_detection_base_model.py) ---

def run_sr_scores(X):
    series = _to_numpy(X)
    if len(series) < 100:
        return statistical_fallback(series)
    try:
        amp_window_size, series_window_size, score_window_size = 36, 36, 50
        safe_score_window = min(score_window_size, len(series) // 3)
        safe_amp_window = min(amp_window_size, len(series) // 4)
        safe_series_window = min(series_window_size, len(series) // 4)
        safe_score_window = max(safe_score_window, 10)
        safe_amp_window = max(safe_amp_window, 5)
        safe_series_window = max(safe_series_window, 5)
        spec = anom.Silency(safe_amp_window, safe_series_window, safe_score_window)
        score = spec.generate_anomaly_score(series)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    except Exception:
        return statistical_fallback(series)

def run_iqr_scores(X, use_log_transform=False):
    series = _to_numpy(X)
    if len(series) < 4: return np.zeros(len(series))
    try:
        if use_log_transform:
            proc_series = np.log1p(series)
        else:
            proc_series = series
        Q1 = np.percentile(proc_series, 25)
        Q3 = np.percentile(proc_series, 75)
        IQR = Q3 - Q1
        if IQR == 0 and not use_log_transform: return np.zeros(len(series))
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomaly_scores = np.maximum(lower_bound - proc_series, proc_series - upper_bound)
        anomaly_scores[anomaly_scores < 0] = 0
        if not use_log_transform and IQR > 0:
            anomaly_scores /= IQR
        return np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)
    except Exception:
        return statistical_fallback(series)

def run_moving_average_scores(X, window_size=7):
    series = _to_numpy(X)
    if len(series) < window_size: return np.zeros(len(series))
    try:
        series_df = pd.Series(series)
        moving_avg = series_df.rolling(window=window_size, center=True, min_periods=1).mean()
        errors = series - moving_avg.values
        error_std = np.std(errors)
        if error_std == 0: return np.zeros(len(series))
        anomaly_scores = np.abs(errors / error_std)
        return np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)
    except Exception:
        return statistical_fallback(series)

def run_iforest_scores(X):
    series = _to_numpy(X)
    if len(series) < 2: return np.zeros(len(series))
    try:
        iforest = IForest()
        iforest.fit(series.reshape(-1, 1))
        scores = iforest.decision_scores_
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    except Exception:
        return statistical_fallback(series)

def run_knn_scores(X):
    series = _to_numpy(X).reshape(-1, 1)
    if len(series) < 2: return np.zeros(len(series))
    try:
        clf = KNN()
        clf.fit(series)
        scores = clf.decision_scores_
        return np.nan_to_num(scores)
    except Exception:
        return statistical_fallback(series.flatten())

def _load_keras_model(model_path, model_var_name):
    global _REPAD_MODEL, _DEEPANT_MODEL
    model_var = globals()[model_var_name]
    if model_var is not None:
        return model_var
    if not os.path.exists(model_path):
        print(f"[{model_var_name}] Không tìm thấy model tại {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        globals()[model_var_name] = model
        return model
    except Exception as e:
        print(f"[{model_var_name}] Lỗi load model: {e}")
        return None

def _run_forecasting_model_scores(X, time_steps, model_loader_func, seq_creator_func):
    series = _to_numpy(X)
    if len(series) < time_steps: return np.zeros(len(series))
    
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).astype(np.float32)
    
    X_seq, y_true = seq_creator_func(series_scaled, time_steps)
    if X_seq.size == 0: return np.zeros(len(series))
    
    model = model_loader_func()
    if model is None: return statistical_fallback(series)
    
    try:
        y_pred = model.predict(X_seq, verbose=0)
        prediction_error = np.abs(y_pred - y_true).flatten()
        scores = np.zeros(len(series))
        scores[time_steps - 1: time_steps - 1 + len(prediction_error)] = prediction_error
        max_score = np.max(scores)
        if max_score > 0: scores /= max_score
        return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception:
        return statistical_fallback(series)

def _create_repad_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:i + time_steps - 1])
        y.append(data[i + time_steps - 1])
    return np.array(X), np.array(y)

def run_repad_scores(X, time_steps=30):
    loader = lambda: _load_keras_model(os.path.join(WEIGHTS_DIR, 'repad_forecasting_model.keras'), '_REPAD_MODEL')
    return _run_forecasting_model_scores(X, time_steps, loader, _create_repad_sequences)

def run_prophet_scores(X):
    series = _to_numpy(X)
    if len(series) < 30: return statistical_fallback(series)
    try:
        df = pd.DataFrame({'ds': pd.to_datetime(pd.to_numeric(np.arange(len(series)), downcast='integer'), unit='D'), 'y': series})
        model = Prophet(interval_width=0.97, seasonality_mode='multiplicative')
        model.fit(df)
        forecast = model.predict(df[['ds']])
        full_df = pd.merge(df, forecast[['ds', 'yhat_lower', 'yhat_upper']], on='ds')
        distance = np.maximum(full_df['y'] - full_df['yhat_upper'], full_df['yhat_lower'] - full_df['y'])
        distance[distance < 0] = 0
        interval_width = full_df['yhat_upper'] - full_df['yhat_lower']
        interval_width[interval_width <= 0] = 1e-8
        scores = distance / interval_width
        return np.nan_to_num(scores.values, nan=0.0, posinf=1e6, neginf=0.0)
    except Exception:
        return statistical_fallback(series)

class _MomentTimeSeriesDataset(Dataset):
    def __init__(self, series, window_size, step):
        self.series = series.astype(np.float32)
        self.window_size = window_size
        self.step = step
    def __len__(self):
        return max(0, (len(self.series) - self.window_size) // self.step + 1)
    def __getitem__(self, idx):
        start_idx = idx * self.step
        end_idx = start_idx + self.window_size
        return torch.tensor(self.series[start_idx:end_idx]).unsqueeze(0)

def _load_moment_model():
    global MOMENT_MODEL, DEVICE
    if MOMENT_MODEL is None:
        try:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            MOMENT_MODEL = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-base", model_kwargs={"task_name": "reconstruction"})
            MOMENT_MODEL.init()
            MOMENT_MODEL = MOMENT_MODEL.to(DEVICE).float()
        except Exception as e:
            MOMENT_MODEL = "failed"
            print(f"Lỗi khi tải mô hình MOMENT: {e}")

def run_moment_scores(X, window_size=128, step_size=32):
    _load_moment_model()
    series = _to_numpy(X)
    if MOMENT_MODEL == "failed" or len(series) < window_size:
        return statistical_fallback(series)
    try:
        dataset = _MomentTimeSeriesDataset(series, window_size, step_size)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        reconstructions = []
        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x.to(DEVICE).float()
                output = MOMENT_MODEL(x_enc=batch_x)
                reconstructions.append(output.reconstruction.detach().cpu().squeeze(-1).numpy())
        if not reconstructions: return statistical_fallback(series)
        reconstructed_windows = np.concatenate(reconstructions, axis=0)
        reconstructed_series = np.zeros_like(series, dtype=float)
        counts = np.zeros_like(series, dtype=float)
        for i, window in enumerate(reconstructed_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            if end_idx <= len(reconstructed_series):
                reconstructed_series[start_idx:end_idx] += window
                counts[start_idx:end_idx] += 1
        counts[counts == 0] = 1
        reconstructed_series /= counts
        reconstructed_series[counts == 0] = series[counts == 0]
        return np.abs(series - reconstructed_series)
    except Exception:
        return statistical_fallback(series)

# --- PHẦN 2: QUY TRÌNH TRỰC QUAN HÓA (ĐÃ SỬA THEO YÊU CẦU) ---

# --- CÁC HẰNG SỐ VÀ CẤU HÌNH ---
DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
MAX_PLACEIDS = 10 
OUTPUT_DIR = 'anomaly_visualizations'
THRESHOLD_METHOD = 'z'

# --- HÀM TIỆN ÍCH CHO TRỰC QUAN HÓA ---
def compute_dynamic_threshold(scores: np.ndarray, method: str = 'z', contamination: float = 0.05) -> float:
    if scores.size == 0: return np.inf
    if method == 'z':
        mean_ = np.mean(scores)
        std_ = np.std(scores)
        if std_ < 1e-9: return mean_ + 1e6
        return mean_ + 2.5 * std_
    elif method == 'percentile':
        return np.percentile(scores, 100 * (1 - contamination))
    else:
        raise ValueError("Phương pháp phải là 'z' hoặc 'percentile'")

def visualize_and_save(series_df, anomaly_indices, place_id, model_name, output_dir):
    """
    Vẽ đồ thị và lưu vào thư mục con được chỉ định.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(series_df.index, series_df.iloc[:, 0], label='Original Time Series', color='royalblue', zorder=1)
    
    if len(anomaly_indices) > 0:
        anomaly_values = series_df.iloc[anomaly_indices, 0]
        plt.scatter(anomaly_indices, anomaly_values, color='red', label='Detected Anomalies', s=50, zorder=2, alpha=0.8)

    plt.title(f'Anomaly Detection for PlaceID: {place_id} using {model_name}')
    plt.xlabel('Time Step')
    plt.ylabel(series_df.columns[0])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Tên file giờ đây chỉ cần tên mô hình vì placeId đã nằm trong tên thư mục
    file_name = f"{model_name}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# --- HÀM CHÍNH ---
def main():
    print("Bắt đầu quá trình trực quan hóa phát hiện bất thường...")

    # Tạo thư mục output chính nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        data_full = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{DATA_FILE_PATH}'.")
        return

    unique_place_ids = data_full['placeId'].unique()
    place_ids_to_process = unique_place_ids[:MAX_PLACEIDS]

    # Định nghĩa các mô hình cần chạy
    model_funcs = {
        'SR': run_sr_scores, 'IQR': run_iqr_scores, 'MA': run_moving_average_scores,
        'IForest': run_iforest_scores, 'KNN': run_knn_scores, 'RePAD': run_repad_scores,
        'Prophet': run_prophet_scores, 'Moment': run_moment_scores
    }

    print(f"\nSẽ xử lý {len(place_ids_to_process)} địa điểm và lưu vào các thư mục con.")
    
    # Vòng lặp qua từng địa điểm
    for place_id in tqdm(place_ids_to_process, desc="Processing PlaceIDs"):
        
        # *** THAY ĐỔI CHÍNH: TẠO THƯ MỤC CON CHO TỪNG PLACEID ***
        place_output_dir = os.path.join(OUTPUT_DIR, f"placeID_{place_id}")
        os.makedirs(place_output_dir, exist_ok=True)

        ts_group = data_full[data_full['placeId'] == place_id].sort_values('date').reset_index(drop=True)
        series_df = ts_group[['view']]

        if series_df.empty:
            print(f"  [WARN] Bỏ qua PlaceID {place_id} vì không có dữ liệu.")
            continue

        # Vòng lặp qua từng mô hình
        for model_name, model_func in model_funcs.items():
            print(f"  Processing: PlaceID {place_id} - Model '{model_name}'")
            try:
                # 1. Tính điểm bất thường
                anomaly_scores = model_func(series_df)
                
                # 2. Xác định ngưỡng và tìm các điểm bất thường
                threshold = compute_dynamic_threshold(anomaly_scores, method=THRESHOLD_METHOD)
                anomaly_indices = np.where(anomaly_scores > threshold)[0]
                
                # 3. Vẽ đồ thị và lưu lại vào THƯ MỤC CON
                visualize_and_save(series_df, anomaly_indices, place_id, model_name, place_output_dir)

            except Exception as e:
                print(f"    -> [LỖI] Mô hình '{model_name}' thất bại cho PlaceID {place_id}. Lỗi: {e}")

    print("\n" + "="*50)
    print("Hoàn thành!")
    print(f"Tất cả đồ thị đã được lưu vào các thư mục con trong: '{OUTPUT_DIR}'")
    print("="*50)

if __name__ == '__main__':
    main()
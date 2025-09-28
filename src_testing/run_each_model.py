import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import t
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import warnings
import sranodec as anom
import joblib
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
import torch
from torch.utils.data import Dataset, DataLoader
from momentfm import MOMENTPipeline
import sys
from dtaidistance import dtw

# --- PHẦN 0: IMPORTS VÀ CÀI ĐẶT ---

# Tắt các cảnh báo không cần thiết để output gọn gàng hơn
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from prophet import Prophet

# Import các metric từ TSAD_eval
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TSAD_DIR = os.path.join(PROJECT_ROOT, 'TSAD_eval')
if TSAD_DIR not in sys.path:
    sys.path.insert(0, TSAD_DIR)
try:
    from TSAD_eval.metrics import pr_from_events
    affiliation_pr = pr_from_events
    print("[INFO] Đã import thành công pr_from_events từ TSAD_eval.")
except ImportError:
    print("[WARN] Không thể import pr_from_events từ thư mục TSAD_eval, sử dụng fallback.")
    affiliation_pr = None

# --- PHẦN 1: METRIC CLASSES IMPLEMENTATION ---

def pointwise_to_segmentwise(pointwise):
    segmentwise = []
    prev = -10
    for point in pointwise:
        if point > prev + 1:
            segmentwise.append([point, point])
        else:
            segmentwise[-1][-1] += 1
        prev = point
    return np.array(segmentwise)

def segmentwise_to_pointwise(segmentwise):
    pointwise = []
    for start, end in segmentwise:
        for point in range(start, end + 1):
            pointwise.append(point)
    return np.array(pointwise)

def pointwise_to_full_series(pointwise, length):
    anomalies_full_series = np.zeros(length)
    if len(pointwise) > 0:
        assert pointwise[-1] < length
        anomalies_full_series[pointwise] = 1
    return anomalies_full_series

def f1_from_pr(p, r, beta=1):
    if r == 0 and p == 0:
        return 0
    return ((1 + beta**2) * r * p) / (beta**2 * p + r)

def recall(*args, tp, fn):
    return 0 if tp + fn == 0 else tp / (tp + fn)

def precision(*args, tp, fp):
    return 0 if tp + fp == 0 else tp / (tp + fp)

class Binary_anomalies:
    def __init__(self, length, anomalies):
        self._length = length
        self._set_anomalies(anomalies)

    def _set_anomalies(self, anomalies):
        anomalies = np.array(anomalies, dtype=int)
        if self._is_pointwise(anomalies):
            anomalies_ptwise = anomalies
            anomalies_segmentwise = pointwise_to_segmentwise(anomalies)
            anomalies_full_series = pointwise_to_full_series(anomalies_ptwise, self._length)
        elif self._is_segmentwise(anomalies):
            anomalies_segmentwise = anomalies
            anomalies_ptwise = segmentwise_to_pointwise(anomalies)
            anomalies_full_series = pointwise_to_full_series(anomalies_ptwise, self._length)
        else:
            raise ValueError(f"Illegal shape of anomalies:\n{anomalies}")

        if len(anomalies_ptwise) > 0:
            assert all(anomalies_ptwise == np.sort(anomalies_ptwise))
            assert anomalies_ptwise[0] >= 0
            assert len(anomalies_ptwise) == len(np.unique(anomalies_ptwise))
            assert len(anomalies_ptwise) == sum(anomalies_full_series)
            if len(anomalies_segmentwise) > 0:
                assert all(anomalies_segmentwise[:, 0] == np.sort(anomalies_segmentwise[:, 0]))
                assert all(anomalies_segmentwise[:, 1] >= anomalies_segmentwise[:, 0])

        self.anomalies_segmentwise = anomalies_segmentwise
        self.anomalies_ptwise = anomalies_ptwise
        self.anomalies_full_series = anomalies_full_series

    def _is_pointwise(self, anomalies):
        return len(anomalies.shape) == 1 and anomalies.shape != (self._length,)

    def _is_segmentwise(self, anomalies):
        return len(anomalies.shape) == 2

    def get_length(self):
        return self._length

class Binary_detection:
    def __init__(self, length, gt_anomalies, predicted_anomalies):
        self._length = length
        self._gt = Binary_anomalies(length, gt_anomalies)
        self._prediction = Binary_anomalies(length, predicted_anomalies)

    def get_length(self): return self._length
    def get_gt_anomalies_ptwise(self): return self._gt.anomalies_ptwise
    def get_gt_anomalies_segmentwise(self): return self._gt.anomalies_segmentwise
    def get_predicted_anomalies_ptwise(self): return self._prediction.anomalies_ptwise
    def get_predicted_anomalies_segmentwise(self): return self._prediction.anomalies_segmentwise
    def get_predicted_anomalies_full_series(self): return self._prediction.anomalies_full_series
    def get_gt_anomalies_full_series(self): return self._gt.anomalies_full_series

class Pointwise_metrics(Binary_detection):
    def __init__(self, *args):
        super().__init__(*args)
        self.set_confusion()

    def set_confusion(self):
        gt = self.get_gt_anomalies_full_series()
        pred = self.get_predicted_anomalies_full_series()
        self.tp = np.sum(pred * gt)
        self.fp = np.sum(pred * (1 - gt))
        self.fn = np.sum((1 - pred) * gt)

class Segmentwise_metrics(Pointwise_metrics):
    def __init__(self, *args):
        super().__init__(*args)
        self.set_confusion()

    def set_confusion(self):
        tp = 0; fn = 0
        for gt_anomaly in self.get_gt_anomalies_segmentwise():
            found = False
            for predicted_anomaly in self.get_predicted_anomalies_segmentwise():
                if self._overlap(gt_anomaly, predicted_anomaly):
                    tp += 1; found = True; break
            if not found: fn += 1
        fp = 0
        for predicted_anomaly in self.get_predicted_anomalies_segmentwise():
            found = False
            for gt_anomaly in self.get_gt_anomalies_segmentwise():
                if self._overlap(gt_anomaly, predicted_anomaly):
                    found = True; break
            if not found: fp += 1
        self.fp = fp; self.fn = fn; self.tp = tp

    def _overlap(self, anomaly1, anomaly2):
        return not (anomaly1[1] < anomaly2[0] or anomaly2[1] < anomaly1[0])

class Redefined_PR_metric(Binary_detection):
    def get_score(self):
        self.r = self.recall()
        self.p = self.precision()
        return f1_from_pr(self.p, self.r)
    def recall(self): raise NotImplementedError
    def precision(self): raise NotImplementedError

class Composite_f(Redefined_PR_metric):
    def __init__(self, *args):
        super().__init__(*args)
        self.pointwise_metrics = Pointwise_metrics(*args)
        self.segmentwise_metrics = Segmentwise_metrics(*args)
    def recall(self):
        return recall(tp=self.segmentwise_metrics.tp, fn=self.segmentwise_metrics.fn)
    def precision(self):
        return precision(tp=self.pointwise_metrics.tp, fp=self.pointwise_metrics.fp)

class Affiliation(Redefined_PR_metric):
    def get_score(self):
        if affiliation_pr is None:
            return self._fallback_affiliation_score()
        
        try:
            pr_output = affiliation_pr(
                self._reformat_segments(self.get_predicted_anomalies_segmentwise()),
                self._reformat_segments(self.get_gt_anomalies_segmentwise()),
                (0, self.get_length()),
            )
            self.r = pr_output["recall"]
            self.p = pr_output["precision"]
            return f1_from_pr(self.p, self.r)
        except Exception as e:
            print(f"[WARN] affiliation_pr failed, using fallback: {e}")
            return self._fallback_affiliation_score()
    
    def _fallback_affiliation_score(self):
        gt_segments = self.get_gt_anomalies_segmentwise()
        pred_segments = self.get_predicted_anomalies_segmentwise()
        
        if len(gt_segments) == 0 and len(pred_segments) == 0:
            return 1.0
        if len(gt_segments) == 0:
            return 0.0
        if len(pred_segments) == 0:
            return 0.0
        
        tp = 0
        for gt_seg in gt_segments:
            for pred_seg in pred_segments:
                if self._overlap(gt_seg, pred_seg):
                    tp += 1
                    break
        
        precision = tp / len(pred_segments) if len(pred_segments) > 0 else 0
        recall = tp / len(gt_segments) if len(gt_segments) > 0 else 0
        return f1_from_pr(precision, recall)
    
    def _overlap(self, seg1, seg2):
        return not (seg1[1] < seg2[0] or seg2[1] < seg1[0])
    
    def _reformat_segments(self, segments):
        return [tuple([start, end + 1]) for start, end in segments]

class Temporal_distance(Binary_detection):
    def get_score(self):
        a = np.array(self.get_gt_anomalies_ptwise())
        b = np.array(self.get_predicted_anomalies_ptwise())
        return self._dist(a, b) + self._dist(b, a)
    def _dist(self, a, b):
        dist = 0
        for pt in a:
            dist += min(abs(b - pt)) if len(b) > 0 else self._length
        return dist

# --- PHẦN 2: CÀI ĐẶT CHUNG VÀ CÁC HÀM MÔ HÌNH ---
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

# ILOF
def _epd(series, j, p):
    """
    Implements the Extreme-Pointed Discriminating (EPD) function from Pseudocode 1.
    Checks if a point is a local maximum, minimum, or neither within a window of size p.
    
    Returns:
        1: if series[j] is a local maximum.
       -1: if series[j] is a local minimum.
        0: otherwise.
    """
    n = len(series)
    # Define the local window, ensuring it's within array bounds
    start = max(0, j - p)
    end = min(n, j + p + 1)
    window = series[start:end]
    
    # The point at index j corresponds to index (j - start) in the window
    local_j_idx = j - start
    
    is_max = True
    is_min = True
    
    # Check against all other points in the window
    for i in range(len(window)):
        if i == local_j_idx:
            continue
        if window[i] > series[j]:
            is_max = False
        if window[i] < series[j]:
            is_min = False
        # Optimization: if it's neither, we can stop early
        if not is_max and not is_min:
            break
            
    if is_max:
        return 1
    elif is_min:
        return -1
    else:
        return 0

def run_ilof_scores(X, max_p=4, k=5):
    """
    Calculates anomaly scores based on the ILOF (Improved Local Outlier Factor) algorithm
    from the paper "Locality-Based Visual Outlier Detection Algorithm for Time Series".

    The algorithm first uses Hierarchy-Segmentation-based Data Extraction (HSDE) to
    identify the most significant extreme points and then calculates a modified Local
    Outlier Factor (LOF) score for these points using Dynamic Time Warping (DTW) as the
    distance metric.

    Args:
        X (pd.DataFrame or pd.Series): The input time series data.
        max_p (int): The maximum hierarchy value to consider in the HSDE step. This defines
                     the largest window for finding extreme points. Corresponds to Max_p
                     in the paper. Defaults to 4, as used in the ECG experiment.
        k (int): The number of nearest neighbors to consider for the LOF calculation.
                 Corresponds to k in k-distance. Defaults to 5.

    Returns:
        np.ndarray: An array of anomaly scores of the same length as the input series.
                    Scores are between 0 and 1. A score closer to 1 indicates a higher
                    likelihood of being an outlier. Points not selected by HSDE have a score of 0.
    """
    series = _to_numpy(X)
    n = len(series)

    # The algorithm requires enough points to form windows and neighborhoods.
    if n < 2 * max_p + 1 or n <= k:
        print(f"Warning (ILOF): Series is too short ({n} points) for the given parameters "
              f"(max_p={max_p}, k={k}). Returning zero scores.")
        return np.zeros(n)

    try:
        # --- Part 1: HSDE (Hierarchy-Segmentation-based Data Extraction) ---
        # This corresponds to Section 3 of the paper.

        # (1.1) HM (Hierarchy Marking of Time Series), based on Pseudocode 2
        hm = np.zeros(n, dtype=int)
        for p in range(1, max_p + 1):
            # Iterate through interior points as per the EPD definition
            for j in range(1, n - 1):
                # Check for local maximum
                if hm[j] == p - 1:
                    if _epd(series, j, p) == 1:
                        hm[j] += 1
                # Check for local minimum (using -(p-1) for negative hierarchy)
                elif hm[j] == -(p - 1):
                     if _epd(series, j, p) == -1:
                        hm[j] -= 1
        
        # Mark endpoints with the highest hierarchy to ensure they act as boundaries
        hm[0] = max_p + 1
        hm[-1] = max_p + 1

        # (1.2) HSSA (Hierarchy Segmentation Series Accessing), based on Pseudocode 3
        # Select points where the absolute hierarchy value is at its maximum.
        # These are the most significant extreme points.
        hss_indices = np.where(np.abs(hm) >= max_p)[0]
        
        if len(hss_indices) <= k:
            print(f"Warning (ILOF): HSDE selected {len(hss_indices)} points, which is not more than k={k}. "
                  "Cannot compute LOF. Returning zero scores.")
            return np.zeros(n)
            
        hss_values = series[hss_indices]

        # --- Part 2: Modified LOF Calculation on HSS ---
        # This corresponds to Section 4 of the paper.

        # (2.1) Calculate the pairwise distance matrix using DTW.
        # The paper specifies DTW as the dissimilarity measurement.
        # We create subsequences of length 1 for the distance calculation.
        hss_subsequences = [np.array([v], dtype=np.float64) for v in hss_values]
        dtw_matrix = dtw.distance_matrix_fast(hss_subsequences, use_c=True)

        # (2.2) Compute k-distance for each object in HSS
        k_distances = np.zeros(len(hss_indices))
        for i in range(len(hss_indices)):
            distances = np.sort(dtw_matrix[i])
            # The k-th nearest neighbor is at index k because index 0 is the point itself (distance 0)
            k_distances[i] = distances[k]

        # (2.3) Compute Local Reachability Density and LOF for each object in HSS
        lof_scores = np.zeros(len(hss_indices))
        for i in range(len(hss_indices)):
            # Find the k-distance neighborhood (N_k)
            # These are all points whose distance is <= k-distance of point i
            neighbor_indices = np.where(dtw_matrix[i] <= k_distances[i])[0]

            # Calculate sum of reachability distances to neighbors
            sum_reach_dist = 0
            for o_idx in neighbor_indices:
                # reach-dist(i, o) = max{k-distance(o), DTW(i, o)}
                reach_dist = max(k_distances[o_idx], dtw_matrix[i][o_idx])
                sum_reach_dist += reach_dist

            # Calculate Local Reachability Density (rho_k) as per Formula (3)
            num_neighbors = len(neighbor_indices)
            if sum_reach_dist > 0:
                rho_k = num_neighbors / sum_reach_dist
            else:
                rho_k = 0 # Avoid division by zero
            
            # Calculate Local Outlier Factor (LOF) as per Formula (4)
            # LOF(x) = 1 - rho_k(x)
            lof_scores[i] = 1 - rho_k

        # --- Part 3: Map Scores Back to Original Series Length ---
        final_scores = np.zeros(n)
        # Assign the calculated LOF scores to the corresponding original indices
        final_scores[hss_indices] = lof_scores
        
        # Normalize to handle potential floating point inaccuracies
        final_scores = np.clip(final_scores, 0, 1)

        return np.nan_to_num(final_scores, nan=0.0)

    except Exception as e:
        print(f"Error during ILOF execution: {e}. Returning zero scores.")
        return np.zeros(n)
    
# SH-ESD+
def run_shesd_scores(X, seasonality=7, alpha=0.05):
    """
    Tính toán điểm bất thường dựa trên thuật toán SH-ESD với cơ chế dừng tự động.
    Thuật toán sẽ tự dừng khi không còn tìm thấy điểm bất thường nào có ý nghĩa thống kê.

    Args:
        X (pd.DataFrame or pd.Series): Chuỗi thời gian đầu vào.
        seasonality (int): Độ dài của một chu kỳ (ví dụ: 7 cho dữ liệu hàng ngày).
        alpha (float): Mức ý nghĩa thống kê để xác định ngưỡng. Giá trị càng nhỏ,
                       điều kiện để một điểm bị coi là bất thường càng khắt khe.

    Returns:
        np.ndarray: Một mảng chứa điểm bất thường. Các điểm không được xác định là
                    bất thường sẽ có điểm số là 0.
    """
    series = _to_numpy(X)
    n = len(series)

    if n < 2 * seasonality:
        print(f"Warning (SH-ESD): Dữ liệu quá ngắn ({n}) cho seasonality ({seasonality}).")
        return statistical_fallback(series)

    try:
        # Bước 1: Phân rã chuỗi thời gian bằng STL
        stl = STL(series, period=seasonality, robust=True)
        result = stl.fit()
        residual = result.resid

        # --- Bước 2: Áp dụng ESD lặp lại với ngưỡng động ---
        anomaly_scores = np.zeros(n)
        
        # Tạo một bản sao của phần dư để có thể "loại bỏ" các điểm bất thường đã tìm thấy
        # Sử dụng mask để theo dõi các điểm chưa bị loại bỏ
        remaining_indices = np.arange(n)
        
        # Giới hạn số lần lặp để tránh vòng lặp vô tận, ví dụ 15% dữ liệu
        max_anoms_to_find = int(np.ceil(n * 0.15))

        for i in range(max_anoms_to_find):
            if len(remaining_indices) < 3: # Cần ít nhất 3 điểm để tính toán
                break

            current_residuals = residual[remaining_indices]
            
            # Tính median và MAD trên các điểm còn lại
            res_median = np.median(current_residuals)
            mad = np.median(np.abs(current_residuals - res_median))
            
            if mad == 0:
                break # Không có độ lệch, không có bất thường

            # Tính điểm thống kê (test statistic) cho các điểm còn lại
            test_statistics = np.abs((current_residuals - res_median) / mad)
            
            # Tìm điểm đáng ngờ nhất
            max_stat_idx_local = np.argmax(test_statistics)
            max_score = test_statistics[max_stat_idx_local]
            
            # --- Tính ngưỡng thống kê động ---
            current_n = len(current_residuals)
            # Dùng phân phối t-student để tìm giá trị tới hạn (critical value)
            p = 1.0 - (alpha / (2.0 * current_n))
            critical_value = t.ppf(p, current_n - 2)
            
            # --- So sánh và Quyết định ---
            if max_score > critical_value:
                # Điểm này là bất thường
                # Lấy chỉ số gốc của điểm bất thường
                original_idx = remaining_indices[max_stat_idx_local]
                
                # Gán điểm số bất thường
                anomaly_scores[original_idx] = max_score
                
                # "Loại bỏ" điểm này khỏi lần xét tiếp theo
                remaining_indices = np.delete(remaining_indices, max_stat_idx_local)
            else:
                # Nếu điểm cao nhất cũng không vượt ngưỡng, dừng tìm kiếm
                break
        
        return np.nan_to_num(anomaly_scores)

    except Exception as e:
        print(f"Lỗi trong quá trình chạy SH-ESD auto: {e}. Sử dụng fallback.")
        return statistical_fallback(series)

# --- PHẦN 2: QUY TRÌNH TRỰC QUAN HÓA (ĐÃ SỬA THEO YÊU CẦU) ---

# --- CÁC HẰNG SỐ VÀ CẤU HÌNH ---
DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
LABEL_DIR = 'new_labels_2'
MAX_PLACEIDS_VISUALIZE = 10  # Chỉ visualize 10 địa điểm đầu
MAX_PLACEIDS_METRICS = 30    # Tính metrics cho 30 địa điểm đầu
OUTPUT_DIR = 'anomaly_visualizations'
THRESHOLD_METHOD = 'percentile'

# --- HÀM TIỆN ÍCH CHO TRỰC QUAN HÓA ---
def compute_dynamic_threshold(scores: np.ndarray, method: str = 'z', contamination: float = 0.02) -> float:
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
    print("Bắt đầu quá trình trực quan hóa phát hiện bất thường và tính toán metrics...")

    # Tạo thư mục output chính nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        data_full = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{DATA_FILE_PATH}'.")
        return

    unique_place_ids = data_full['placeId'].unique()
    place_ids_for_visualization = unique_place_ids[:MAX_PLACEIDS_VISUALIZE]
    place_ids_for_metrics = unique_place_ids[:MAX_PLACEIDS_METRICS]

    # Định nghĩa các mô hình cần chạy
    model_funcs = {
        'SR': run_sr_scores, 
        'MA': run_moving_average_scores,
        'IForest': run_iforest_scores, 
        'KNN': run_knn_scores, 
        'RePAD': run_repad_scores,
        'Prophet': run_prophet_scores, 
        'Moment': run_moment_scores, 
        'SH-ESD': run_shesd_scores,
    }

    print(f"\nSẽ visualize {len(place_ids_for_visualization)} địa điểm và tính metrics cho {len(place_ids_for_metrics)} địa điểm.")
    
    # List để lưu kết quả metrics
    all_metrics_results = []
    
    # Vòng lặp qua từng địa điểm để tính metrics
    for place_id in tqdm(place_ids_for_metrics, desc="Processing PlaceIDs for Metrics"):
        
        ts_group = data_full[data_full['placeId'] == place_id].sort_values('date').reset_index(drop=True)
        series_df = ts_group[['view']]

        if series_df.empty:
            print(f"  [WARN] Bỏ qua PlaceID {place_id} vì không có dữ liệu.")
            continue

        # Đọc nhãn thật từ file labels
        label_path = os.path.join(LABEL_DIR, f'label_{place_id}.csv')
        try:
            label_df = pd.read_csv(label_path)
            true_labels = label_df['label'].to_numpy()
        except FileNotFoundError:
            print(f"  [WARN] Không tìm thấy nhãn cho PlaceID {place_id}, bỏ qua.")
            continue

        n = len(series_df)
        if len(true_labels) > n:
            true_labels = true_labels[:n]
        elif len(true_labels) < n:
            true_labels = np.pad(true_labels, (0, n - len(true_labels)), 'constant', constant_values=0)

        # Chỉ tạo thư mục con cho visualization nếu nằm trong top 10
        should_visualize = place_id in place_ids_for_visualization
        if should_visualize:
            place_output_dir = os.path.join(OUTPUT_DIR, f"placeID_{place_id}")
            os.makedirs(place_output_dir, exist_ok=True)

        # Vòng lặp qua từng mô hình
        for model_name, model_func in model_funcs.items():
            print(f"  Processing: PlaceID {place_id} - Model '{model_name}'")
            try:
                # 1. Tính điểm bất thường
                anomaly_scores = model_func(series_df)
                
                # 2. Xác định ngưỡng và tìm các điểm bất thường
                threshold = compute_dynamic_threshold(anomaly_scores, method=THRESHOLD_METHOD)
                anomaly_indices = np.where(anomaly_scores > threshold)[0]
                predicted_labels = np.zeros(n, dtype=int)
                predicted_labels[anomaly_indices] = 1
                
                # 3. Tính các metrics
                y_true_pt = np.where(true_labels > 0)[0]
                y_pred_pt = np.where(predicted_labels > 0)[0]
                
                comp_f1 = Composite_f(n, y_true_pt, y_pred_pt).get_score()
                aff_f1 = Affiliation(n, y_true_pt, y_pred_pt).get_score()
                td = Temporal_distance(n, y_true_pt, y_pred_pt).get_score()
                
                # Lưu kết quả metrics
                all_metrics_results.append({
                    'placeId': place_id,
                    'model': model_name,
                    'CompF1': comp_f1,
                    'AffF1': aff_f1,
                    'TD': td
                })
                
                # 4. Vẽ đồ thị chỉ cho 10 địa điểm đầu
                if should_visualize:
                    visualize_and_save(series_df, anomaly_indices, place_id, model_name, place_output_dir)

            except Exception as e:
                print(f"    -> [LỖI] Mô hình '{model_name}' thất bại cho PlaceID {place_id}. Lỗi: {e}")
                # Thêm kết quả lỗi vào metrics
                all_metrics_results.append({
                    'placeId': place_id,
                    'model': model_name,
                    'CompF1': 0.0,
                    'AffF1': 0.0,
                    'TD': float('inf')
                })

    # Lưu kết quả metrics vào file CSV
    if all_metrics_results:
        metrics_df = pd.DataFrame(all_metrics_results)
        metrics_csv_path = os.path.join(OUTPUT_DIR, 'all_models_metrics_results.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        print("\n" + "="*80)
        print("KẾT QUẢ METRICS TỔNG HỢP")
        print("="*80)
        
        # Tính trung bình theo từng mô hình
        avg_by_model = metrics_df.groupby('model')[['CompF1', 'AffF1', 'TD']].mean()
        print("Trung bình theo mô hình:")
        print(avg_by_model.round(4))
        
        print(f"\nĐã lưu tất cả kết quả metrics vào: {metrics_csv_path}")
        print("="*80)

    print("\n" + "="*50)
    print("Hoàn thành!")
    print(f"Tất cả đồ thị visualizations đã được lưu vào các thư mục con trong: '{OUTPUT_DIR}'")
    print(f"Metrics đã được tính cho {MAX_PLACEIDS_METRICS} địa điểm và visualizations cho {MAX_PLACEIDS_VISUALIZE} địa điểm")
    print("="*50)

if __name__ == '__main__':
    main()
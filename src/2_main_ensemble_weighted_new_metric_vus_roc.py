import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Bảo đảm root project nằm trong sys.path để import TSAD_eval
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util
import traceback
affiliation_pr = None

# Thử import TSAD_eval.metrics
try:
    # Thêm TSAD_eval directory vào sys.path trước
    TSAD_DIR = os.path.join(PROJECT_ROOT, 'TSAD_eval')
    if TSAD_DIR not in sys.path:
        sys.path.insert(0, TSAD_DIR)
    
    # Thử import trực tiếp từ TSAD_eval package
    import TSAD_eval.metrics as _metrics_mod    # type: ignore
    affiliation_pr = getattr(_metrics_mod, 'pr_from_events', None)
    print('[INFO] Đã import TSAD_eval.metrics thành công.')
except Exception as e_pkg:
    print(f"[WARN] Không import được TSAD_eval.metrics: {e_pkg}")
    # Fallback: import metrics.py trực tiếp từ thư mục TSAD_eval
    try:
        import metrics as _metrics_mod    # type: ignore
        affiliation_pr = getattr(_metrics_mod, 'pr_from_events', None)
        print('[INFO] Đã import metrics.py bằng fallback mode.')
    except Exception as e_fb:
        print(f"[ERROR] Fallback import metrics.py cũng thất bại: {e_fb}")
        print('[INFO] Sẽ sử dụng fallback affiliation score implementation.')
        affiliation_pr = None  # Set to None when import fails

# Tắt TensorFlow / các cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import anomaly_detection_base_model
import anomaly_injection_2 as anomaly_injection
import model_centrality
import get_prediction_error_ranking
import borda_count_rank_aggregation


# --- Sao chép toàn bộ mã nguồn cần thiết từ metrics.py vào đây ---
# Helper functions and core classes
def pointwise_to_segmentwise(pointwise):
    """Reformat anomaly time series from pointwise to segmentwise"""
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
    """Reformat anomaly time series from segmentwise to pointwise"""
    pointwise = []
    for start, end in segmentwise:
        for point in range(start, end + 1):
            pointwise.append(point)
    return np.array(pointwise)

def pointwise_to_full_series(pointwise, length):
    """Reformat anomaly time series from pointwise to full_series"""
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

def f1_score(*args, tp, fp, fn, beta=1):
    r = recall(tp=tp, fn=fn)
    p = precision(tp=tp, fp=fp)
    return f1_from_pr(p, r, beta=beta)

class Binary_anomalies:
    def __init__(self, length, anomalies):
        self._length = length
        self._set_anomalies(anomalies)

    def _set_anomalies(self, anomalies):
        # ĐẢM BẢO indices là kiểu int (tránh object gây lỗi IndexError khi indexing)
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

# Metric-specific classes
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

    def get_score(self):
        return f1_score(tp=self.tp, fn=self.fn, fp=self.fp)

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
            # Fallback: use simple segment-based F1 when TSAD_eval is not available
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
        # Simple segment-based F1 as fallback
        gt_segments = self.get_gt_anomalies_segmentwise()
        pred_segments = self.get_predicted_anomalies_segmentwise()
        
        if len(gt_segments) == 0 and len(pred_segments) == 0:
            return 1.0
        if len(gt_segments) == 0:
            return 0.0
        if len(pred_segments) == 0:
            return 0.0
        
        # Count overlapping segments
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


# --- Additional functions for VUS-ROC ---
def get_anomaly_segments(labels):
    segments = []
    start = None
    for i in range(len(labels)):
        if labels[i] > 0:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments

def compute_r_auc_roc(score, labels, ell):
    length = len(labels)
    if length == 0:
        return 0.0
    sorted_indices = np.argsort(-score)
    rank = np.empty(length, dtype=int)
    rank[sorted_indices] = np.arange(length)
    segments = get_anomaly_segments(labels)
    num_R = len(segments)
    min_ranks = []
    for s, e in segments:
        left = max(0, s - (ell // 2))
        right = min(length - 1, e + (ell // 2))
        extended = range(left, right + 1)
        min_rank = min(rank[i] for i in extended) if extended else length
        min_ranks.append(min_rank)
    min_ranks = np.array(min_ranks)
    label = labels.astype(float)
    label_ell = np.zeros(length)
    for s, e in segments:
        for i in range(s, e + 1):
            label_ell[i] = 1
        left = s - (ell // 2)
        for i in range(max(0, left), s):
            dist = s - i
            label_ell[i] = max(label_ell[i], (1 - dist / ell) ** 0.5 if ell > 0 else 0)
        right = e + (ell // 2)
        for i in range(e + 1, min(length, right + 1)):
            dist = i - e
            label_ell[i] = max(label_ell[i], (1 - dist / ell) ** 0.5 if ell > 0 else 0)
    P_ell = (np.sum(label) + np.sum(label_ell)) / 2
    N_ell = length - P_ell
    cum_TP = np.cumsum(label_ell[sorted_indices])
    cum_FP = np.cumsum(1 - label_ell[sorted_indices])
    TPR = np.zeros(length + 1)
    FPR = np.zeros(length + 1)
    for k in range(length + 1):
        num_detected = np.sum(min_ranks < k) if num_R > 0 else 0
        bonus = num_detected / num_R if num_R > 0 else 1
        TPR[k] = cum_TP[k - 1] / P_ell * bonus if P_ell > 0 and k > 0 else 0
        FPR[k] = cum_FP[k - 1] / N_ell if N_ell > 0 and k > 0 else 0
    if P_ell > 0:
        TPR[-1] = min(1.0, TPR[-1])
    auc = 0.0
    for k in range(1, length + 1):
        dFPR = FPR[k] - FPR[k - 1]
        avg_TPR = (TPR[k] + TPR[k - 1]) / 2
        auc += avg_TPR * dFPR
    return auc

def compute_vus_roc(score, labels, max_ell=100, num_ell=20):
    if np.sum(labels) == 0 or np.std(score) < 1e-9:
        return 0.0
    ells = np.linspace(0, max_ell, num_ell, dtype=int)
    aucs = [compute_r_auc_roc(score, labels, ell) for ell in ells]
    vus = 0.0
    for i in range(1, num_ell):
        d = ells[i] - ells[i - 1]
        avg_auc = (aucs[i] + aucs[i - 1]) / 2
        vus += avg_auc * d
    vus /= max_ell if max_ell > 0 else 1
    return vus


# --- Phần mã nguồn chính ---
def normalize_scores(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0: return arr
    std = np.std(arr)
    if std < 1e-9: return np.zeros_like(arr)
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-12: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def derive_weights_from_rankings(ranking_lists, models_order):
    raw_scores = {m: 0.0 for m in models_order}
    for ranking in ranking_lists:
        for i, m in enumerate(ranking):
            if m in raw_scores:
                raw_scores[m] += 1.0 / (i + 1.0)
    total = sum(raw_scores.values())
    if total <= 0:
        equal = 1.0 / len(models_order)
        return {m: equal for m in models_order}, raw_scores
    weights = {m: v / total for m, v in raw_scores.items()}
    return weights, raw_scores

def weighted_aggregate(scores_dict: dict, weights: dict) -> np.ndarray:
    intersect = [m for m in weights.keys() if m in scores_dict]
    if not intersect: raise RuntimeError("Không có model nào để tổng hợp.")
    first_len = len(next(iter(scores_dict.values())))
    agg = np.zeros(first_len)
    total_w = 0.0
    for m in intersect:
        s = scores_dict[m]
        if len(s) != first_len:
            fixed = np.zeros(first_len)
            L = min(len(s), first_len)
            fixed[:L] = s[:L]
            s = fixed
        w = weights.get(m, 0.0)
        agg += w * s
        total_w += w
    if total_w <= 1e-12:
        eq_w = 1.0 / len(intersect)
        agg = np.zeros(first_len)
        for m in intersect:
            agg += eq_w * scores_dict[m]
    return agg

def compute_dynamic_threshold(scores: np.ndarray, method: str = 'z', contamination: float = 0.05) -> float:
    if scores.size == 0: return 0.0
    if method == 'z':
        mean_ = np.mean(scores)
        std_ = np.std(scores)
        if std_ < 1e-9: return mean_ + 1e6
        return mean_ + 2.5 * std_
    elif method == 'percentile':
        return np.percentile(scores, 100 * (1 - contamination))
    else:
        raise ValueError("method phải là 'z' hoặc 'percentile'")


if __name__ == '__main__':
    DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'labels/'
    MAX_PLACEIDS = 30
    THRESH_METHOD = 'z'
    CONTAMINATION = 0.05
    WEIGHT_STRATEGY = 'harmonic' # harmonic là derive_weights_from_rankings
    DEBUG_MODE = True

    print("Đang đọc dữ liệu...")
    data_full = pd.read_csv(DATA_FILE_PATH)
    unique_place_ids = data_full['placeId'].unique()
    place_ids_to_process = unique_place_ids[:MAX_PLACEIDS]
    data_subset = data_full[data_full['placeId'].isin(place_ids_to_process)]

    print("\nBắt đầu Single-Judge Weighted Ensemble: VUS-ROC Synthetic...")
    evaluation_results = []

    model_funcs = {
        'SR': anomaly_detection_base_model.run_sr_scores, 
        'IQR': anomaly_detection_base_model.run_iqr_scores,
        'MA': anomaly_detection_base_model.run_moving_average_scores,
        'IForest': anomaly_detection_base_model.run_iforest_scores,
        'KNN': anomaly_detection_base_model.run_knn_scores,
        'RePAD': anomaly_detection_base_model.run_repad_scores,
        'Prophet': anomaly_detection_base_model.run_prophet_scores,
        'Moment': anomaly_detection_base_model.run_moment_scores
    }
    model_names = list(model_funcs.keys())

    for place_id in tqdm(place_ids_to_process, desc="Processing PlaceIDs"):
        ts_group = data_subset[data_subset['placeId'] == place_id].sort_values('date')
        series_df = ts_group[['view']]
        
        # Load nhãn thật để dùng ở cuối
        label_path = os.path.join(LABELS_DIR, f'label_{place_id}.csv')
        if not os.path.exists(label_path): continue
        df_label = pd.read_csv(label_path)
        if 'label' not in df_label.columns: continue
        y_true = (df_label['label'].fillna(0).astype(float) > 0).astype(int).to_numpy()
        if len(series_df) != len(y_true): continue
        y_true_pt = np.where(y_true > 0)[0]
        length = len(y_true)

        # Chạy tất cả model trên dữ liệu thật MỘT LẦN để lấy raw_scores
        raw_scores_real_data = {}
        for m, fn in model_funcs.items():
            try:
                s = fn(series_df)
            except Exception:
                s = np.zeros(length)
            raw_scores_real_data[m] = s

        # --- GIAI ĐOẠN 1: TÌM TRỌNG SỐ BẰNG NHÃN GIẢ ---
        ts_np = series_df['view'].to_numpy()
        
        # Judge 1: VUS-ROC trên synthetic anomaly injection
        injection_tests = {
            'spike': anomaly_injection.inject_spike_anomaly,
            # 'flip': anomaly_injection.inject_flip_anomaly,
            # 'noise': anomaly_injection.inject_noise_anomaly,
            # 'scale': anomaly_injection.inject_scale_anomaly,
            'contextual': anomaly_injection.inject_contextual_anomaly,
            # 'speedup': anomaly_injection.inject_speedup_anomaly,
            'cutoff': anomaly_injection.inject_cutoff_anomaly,
            'wander': anomaly_injection.inject_wander_anomaly,
            'average': anomaly_injection.inject_average_anomaly
        }
        
        metrics_per_test = {}
        for test_name, inject_func in injection_tests.items():
            injected_series, pseudo_labels = inject_func(ts_np)
            if len(np.unique(pseudo_labels)) < 2: continue
            
            length_inj = len(pseudo_labels)
            test_metrics = {}
            injected_df = pd.DataFrame(injected_series)
            
            for m_name, fn in model_funcs.items():
                try: s_inj = fn(injected_df)
                except Exception: s_inj = np.zeros(length_inj)
                
                s_norm = normalize_scores(np.asarray(s_inj))
                
                vus_score_inj = compute_vus_roc(s_norm, pseudo_labels, max_ell=length_inj//10)
                
                test_metrics[m_name] = vus_score_inj
            metrics_per_test[test_name] = test_metrics

        # Tính VUS-ROC trung bình và ranking
        synth_metric_avg = {m: np.nanmean([metrics_per_test.get(t, {}).get(m, np.nan) for t in metrics_per_test]) for m in model_names}
        valid_models_vus = [m for m in model_names if not np.isnan(synth_metric_avg[m])]
        vus_ranking = sorted(valid_models_vus, key=lambda m: synth_metric_avg[m], reverse=True)
        
        # --- BƯỚC LỌC MỚI ---
        VUS_THRESHOLD = 0.6

        # Lọc các model dựa trên VUS-ROC
        eligible_models_vus = {m for m, score in synth_metric_avg.items() if score > VUS_THRESHOLD}

        # Sử dụng chỉ VUS-ROC để lọc
        eligible_models = list(eligible_models_vus)

        # Nếu không có model nào đủ điều kiện, hãy dùng fallback (ví dụ: chỉ dùng top 3 của VUS)
        if not eligible_models:
            print("  [WARN] Không có model nào vượt qua ngưỡng VUS. Fallback về top 3 của VUS.")
            eligible_models = vus_ranking[:3]
            if not eligible_models: # Trường hợp tệ nhất
                eligible_models = model_names

        print(f"  Các model đủ điều kiện tham gia ensemble: {eligible_models}")

        # Lọc lại bảng xếp hạng để chỉ chứa các model đủ điều kiện
        vus_ranking_filtered = [m for m in vus_ranking if m in eligible_models]

        # Tính trọng số CHỈ dựa trên VUS-ROC ranking
        ranking_lists = [vus_ranking_filtered]
        weights, raw_rank_scores = derive_weights_from_rankings(ranking_lists, eligible_models)

        # Tổng hợp điểm số CHỈ từ các model đã được lọc
        normalized_scores_filtered = {m: normalize_scores(raw_scores_real_data[m]) for m in eligible_models}
        aggregated_scores = weighted_aggregate(normalized_scores_filtered, weights)
        threshold_ens = compute_dynamic_threshold(aggregated_scores, method=THRESH_METHOD, contamination=CONTAMINATION)
        y_pred_ens = (aggregated_scores > threshold_ens).astype(int)
        y_pred_pt_ens = np.where(y_pred_ens > 0)[0]

        try: comp_f_score = Composite_f(length, y_true_pt, y_pred_pt_ens).get_score()
        except Exception: comp_f_score = np.nan
        try: aff_f_score = Affiliation(length, y_true_pt, y_pred_pt_ens).get_score()
        except Exception: aff_f_score = np.nan
        try: td_score = Temporal_distance(length, y_true_pt, y_pred_pt_ens).get_score()
        except Exception: td_score = np.nan
        
        print(f"  Ensemble Metrics on TRUE labels: CompF1={comp_f_score:.3f}, AffF1={aff_f_score:.3f}, TD={td_score:.1f}")

        evaluation_results.append({
            'placeId': place_id,
            'ensemble_comp_f1': comp_f_score,
            'ensemble_aff_f1': aff_f_score,
            'ensemble_td': td_score,
            'num_true': int(y_true.sum()),
            'ensemble_num_pred': int(y_pred_ens.sum()),
            'len': length,
            'vus_avg': synth_metric_avg.get(max(weights, key=weights.get), np.nan),
            'top_model_vus': max(weights, key=weights.get)
        })

    # --- TỔNG KẾT KẾT QUẢ ---
    print("\n" + "=" * 80)
    print("KẾT QUẢ SINGLE-JUDGE WEIGHTED ENSEMBLE (VUS-ROC)")
    print("=" * 80)
    if not evaluation_results:
        print("Không có kết quả.")
    else:
        df_res = pd.DataFrame(evaluation_results)
        for _, r in df_res.iterrows():
            print(f"PlaceID {r.placeId}: Ens_CompF1={r.ensemble_comp_f1:.3f} | Ens_AffF1={r.ensemble_aff_f1:.3f} | Ens_TD={r.ensemble_td:.1f} | TopVUS={r.top_model_vus} | True={r.num_true} Pred={r.ensemble_num_pred}")
        print("-" * 80)
        
        avg_comp_f1 = df_res.ensemble_comp_f1.mean()
        avg_aff_f1 = df_res.ensemble_aff_f1.mean()
        avg_td = df_res.ensemble_td.mean()
        
        print(f"AVERAGE Single-Judge Ensemble CompF1: {avg_comp_f1:.4f} trên {len(df_res)} placeId")
        print(f"AVERAGE Single-Judge Ensemble AffF1: {avg_aff_f1:.4f}")
        print(f"AVERAGE Single-Judge Ensemble TD: {avg_td:.4f} (càng thấp càng tốt)")

        out_csv = 'results_single_judge_ensemble_vus.csv'
        df_res.to_csv(out_csv, index=False)
        summary_path = 'results_single_judge_ensemble_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('TÓM TẮT SINGLE-JUDGE ENSEMBLE (VUS-ROC)\n')
            f.write('=' * 60 + '\n')
            f.write(f'Số placeId đã xử lý: {len(df_res)}\n\n')
            f.write(f'--- Single-Judge Ensemble Composite F1-Score ---\n  Trung bình: {avg_comp_f1:.4f}\n')
            f.write(f'--- Single-Judge Ensemble Affiliation F1-Score ---\n  Trung bình: {avg_aff_f1:.4f}\n')
            f.write(f'--- Single-Judge Ensemble Temporal Distance ---\n  Trung bình: {avg_td:.4f}\n\n')
            f.write('Judge: VUS-ROC trên synthetic anomaly injection\n')
            f.write('Aggregation: Harmonic mean của rankings (1/rank)\n')
        print(f"\nĐã lưu kết quả chi tiết: {out_csv} và summary: {summary_path}")
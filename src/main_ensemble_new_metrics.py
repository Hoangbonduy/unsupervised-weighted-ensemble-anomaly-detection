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

import importlib, traceback
affiliation_pr = None
try:
    spec_metrics = importlib.util.find_spec('TSAD_eval.metrics')
    print(f"[DEBUG] spec TSAD_eval.metrics = {spec_metrics}")
    import TSAD_eval.metrics as _metrics_mod    # type: ignore
    affiliation_pr = getattr(_metrics_mod, 'pr_from_events', None)
except Exception as e_pkg:
    print(f"[WARN] Không import được dạng package: {e_pkg}")
    traceback.print_exc(limit=1)
    # Fallback: thêm trực tiếp thư mục TSAD_eval vào path và import kiểu module thường
    TSAD_DIR = os.path.join(PROJECT_ROOT, 'TSAD_eval')
    if TSAD_DIR not in sys.path:
        sys.path.insert(0, TSAD_DIR)
    try:
        import metrics as _metrics_mod    # type: ignore
        affiliation_pr = getattr(_metrics_mod, 'pr_from_events', None)
        print('[INFO] Đã import TSAD_eval bằng fallback directory mode.')
    except Exception as e_fb:
        print(f"[ERROR] Fallback import cũng thất bại: {e_fb}")
        traceback.print_exc(limit=1)
        print('[HINT] Kiểm tra file metrics.py có lỗi cú pháp hoặc tên file khác thường, và xóa __pycache__.')

# Tắt TensorFlow / các cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import anomaly_detection_base_model
import anomaly_injection
import model_centrality
import get_prediction_error_ranking
import borda_count_rank_aggregation


# --- THAY ĐỔI 2: Sao chép toàn bộ mã nguồn cần thiết từ metrics.py vào đây ---

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
        pr_output = affiliation_pr(
            self._reformat_segments(self.get_predicted_anomalies_segmentwise()),
            self._reformat_segments(self.get_gt_anomalies_segmentwise()),
            (0, self.get_length()),
        )
        self.r = pr_output["recall"]
        self.p = pr_output["precision"]
        return f1_from_pr(self.p, self.r)
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


# --- Phần mã nguồn chính của bạn ---

def normalize_scores(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    std = np.std(arr)
    if std < 1e-9:
        return np.zeros_like(arr)
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
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


def derive_weights_from_borda(borda_scores: dict):
    total = sum(borda_scores.values())
    if total <= 0:
        n = len(borda_scores)
        return {m: 1.0 / n for m in borda_scores}
    return {m: s / total for m, s in borda_scores.items()}


def weighted_aggregate(scores_dict: dict, weights: dict) -> np.ndarray:
    intersect = [m for m in weights.keys() if m in scores_dict]
    if not intersect:
        raise RuntimeError("Không có model nào để tổng hợp.")
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
    if scores.size == 0:
        return 0.0
    if method == 'z':
        mean_ = np.mean(scores)
        std_ = np.std(scores)
        if std_ < 1e-9:
            return mean_ + 1e6
        return mean_ + 2.5 * std_
    elif method == 'percentile':
        return np.percentile(scores, 100 * (1 - contamination))
    else:
        raise ValueError("method phải là 'z' hoặc 'percentile'")


if __name__ == '__main__':
    print("affiliation_pr=", affiliation_pr)

    DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'labels/'
    MAX_PLACEIDS = 30
    THRESH_METHOD = 'z'
    CONTAMINATION = 0.05
    WEIGHT_STRATEGY = 'harmonic'
    DEBUG_MODE = True  # Bật để in thêm thông tin chẩn đoán khi lỗi metric

    print("Đang đọc dữ liệu...")
    try:
        data_full = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Không tìm thấy file dữ liệu: {DATA_FILE_PATH}")

    unique_place_ids = data_full['placeId'].unique()
    place_ids_to_process = unique_place_ids[:MAX_PLACEIDS]
    data_subset = data_full[data_full['placeId'].isin(place_ids_to_process)]

    print("\nBắt đầu chọn mô hình bằng nhãn GIẢ (anomaly injection) rồi đánh giá trên nhãn THẬT với CompF1, AffF1, TD ...")
    evaluation_results = []

    model_funcs = {
        'SR': anomaly_detection_base_model.run_sr_scores,
        'IQR': anomaly_detection_base_model.run_iqr_scores,
        'MA': anomaly_detection_base_model.run_moving_average_scores,
        'IForest': anomaly_detection_base_model.run_iforest_scores,
        'KNN': anomaly_detection_base_model.run_knn_scores,
        'RePAD': anomaly_detection_base_model.run_repad_scores,
        'Prophet': anomaly_detection_base_model.run_prophet_scores,
    }
    model_names = list(model_funcs.keys())

    for place_id in tqdm(place_ids_to_process, desc="Processing PlaceIDs"):
        ts_group = data_subset[data_subset['placeId'] == place_id].sort_values('date')
        series_df = ts_group[['view']]

        label_path = os.path.join(LABELS_DIR, f'label_{place_id}.csv')
        if not os.path.exists(label_path):
            continue
        df_label = pd.read_csv(label_path)
        # Chuẩn hóa nhãn về nhị phân 0/1 (bất kỳ giá trị >0 coi là anomalous)
        if 'label' not in df_label.columns:
            if DEBUG_MODE:
                print(f"[WARN] File label không có cột 'label': {label_path}. Columns={df_label.columns.tolist()}")
            continue
        # Ép kiểu và xử lý NaN
        y_true_raw = df_label['label'].fillna(0)
        try:
            y_true = (y_true_raw.astype(float) > 0).astype(int).to_numpy()
        except Exception as e_cast:
            if DEBUG_MODE:
                print(f"[ERROR] Không ép được nhãn sang float tại {label_path}: {e_cast}")
            continue
        if len(series_df) != len(y_true):
            if DEBUG_MODE:
                print(f"[WARN] Độ dài chuỗi và nhãn khác nhau placeId={place_id}: ts={len(series_df)} labels={len(y_true)} -> bỏ qua")
            continue

        raw_scores = {}
        for m, fn in model_funcs.items():
            try:
                s = fn(series_df)
            except Exception as e:
                s = np.zeros(len(series_df))
            raw_scores[m] = s

        # 1) CHỌN MÔ HÌNH DỰA TRÊN NHÃN GIẢ nhưng dùng 3 METRIC (CompF1, AffF1, TD) để đánh giá và RANKING
        #    Thay vì dùng AUC như trước, ta tiêm nhiều loại bất thường, tính 3 metric trên pseudo-labels cho mỗi model.
        #    Sau đó rank từng metric rồi tổng hợp bằng điểm harmonic (1/rank) để chọn best_model.
        ts_np = series_df['view'].to_numpy()

        injection_tests = {
            'spike': anomaly_injection.inject_spike_anomaly,
            'level_shift': anomaly_injection.inject_level_shift_anomaly,
            'flip': anomaly_injection.inject_flip_anomaly,
            'noise': anomaly_injection.inject_noise_anomaly,
            'dip': anomaly_injection.inject_dip_anomaly,
        }

        # Lưu metric theo dạng: metrics_per_test[test_name][model_name] = {'comp':..,'aff':..,'td':..}
        metrics_per_test = {}
        for test_name, inject_func in injection_tests.items():
            injected_series, pseudo_labels = inject_func(ts_np)
            if len(np.unique(pseudo_labels)) < 2:
                # Không đủ lớp để meaningful, bỏ qua test này
                if DEBUG_MODE:
                    print(f"  [INFO] Bỏ qua test {test_name} vì pseudo_labels chỉ có 1 lớp")
                continue
            pseudo_pt = np.where(pseudo_labels > 0)[0]
            length_inj = len(pseudo_labels)
            test_metrics = {}
            injected_df = pd.DataFrame(injected_series)
            for m_name, fn in model_funcs.items():
                try:
                    s_inj = fn(injected_df)
                except Exception as e_model_inj:
                    if DEBUG_MODE:
                        print(f"    [WARN] Model {m_name} lỗi trên test {test_name}: {e_model_inj}")
                    s_inj = np.zeros(length_inj)
                s_norm = normalize_scores(np.asarray(s_inj))
                thr_inj = compute_dynamic_threshold(s_norm, method=THRESH_METHOD, contamination=CONTAMINATION)
                y_pred_inj = (s_norm > thr_inj).astype(int)
                y_pred_pt_inj = np.where(y_pred_inj > 0)[0]
                # Tính metrics trên pseudo labels
                try:
                    comp_score_inj = Composite_f(length_inj, pseudo_pt, y_pred_pt_inj).get_score()
                except Exception as e_comp_inj:
                    if DEBUG_MODE:
                        print(f"      [ERR] Composite_f lỗi test={test_name} model={m_name}: {e_comp_inj}")
                    comp_score_inj = np.nan
                try:
                    aff_score_inj = Affiliation(length_inj, pseudo_pt, y_pred_pt_inj).get_score()
                except Exception as e_aff_inj:
                    if DEBUG_MODE:
                        print(f"      [ERR] Affiliation lỗi test={test_name} model={m_name}: {e_aff_inj}")
                    aff_score_inj = np.nan
                try:
                    td_score_inj = Temporal_distance(length_inj, pseudo_pt, y_pred_pt_inj).get_score()
                except Exception as e_td_inj:
                    if DEBUG_MODE:
                        print(f"      [ERR] Temporal_distance lỗi test={test_name} model={m_name}: {e_td_inj}")
                    td_score_inj = np.nan
                test_metrics[m_name] = {
                    'comp': comp_score_inj,
                    'aff': aff_score_inj,
                    'td': td_score_inj,
                }
            metrics_per_test[test_name] = test_metrics

        # Gộp metric trung bình qua các test cho mỗi model
        synth_metric_avg = {m: {'comp': [], 'aff': [], 'td': []} for m in model_names}
        for test_name, test_metrics in metrics_per_test.items():
            for m in model_names:
                if m in test_metrics:
                    vals = test_metrics[m]
                    synth_metric_avg[m]['comp'].append(vals['comp'])
                    synth_metric_avg[m]['aff'].append(vals['aff'])
                    synth_metric_avg[m]['td'].append(vals['td'])
        # Lấy mean (bỏ NaN)
        for m in model_names:
            for k in ['comp', 'aff', 'td']:
                arr = np.array(synth_metric_avg[m][k], dtype=float)
                arr = arr[~np.isnan(arr)]
                synth_metric_avg[m][k] = float(arr.mean()) if arr.size > 0 else np.nan

        # Ranking theo từng metric (CompF1 & AffF1: lớn hơn tốt, TD: nhỏ hơn tốt)
        # Loại bỏ model toàn NaN (giữ thứ tự còn lại)
        valid_models_comp = [m for m in model_names if not np.isnan(synth_metric_avg[m]['comp'])]
        valid_models_aff = [m for m in model_names if not np.isnan(synth_metric_avg[m]['aff'])]
        valid_models_td = [m for m in model_names if not np.isnan(synth_metric_avg[m]['td'])]
        comp_rank = sorted(valid_models_comp, key=lambda m: synth_metric_avg[m]['comp'], reverse=True)
        aff_rank = sorted(valid_models_aff, key=lambda m: synth_metric_avg[m]['aff'], reverse=True)
        td_rank = sorted(valid_models_td, key=lambda m: synth_metric_avg[m]['td'])

        ranking_lists = [comp_rank, aff_rank, td_rank]
        weights_harmonic, raw_rank_scores = derive_weights_from_rankings(ranking_lists, model_names)
        # best model là model có weight lớn nhất
        best_model = max(weights_harmonic.items(), key=lambda kv: kv[1])[0]

        print("  Synthetic metric averages (pseudo labels):")
        for m in model_names:
            sm = synth_metric_avg[m]
            print(f"    {m}: Comp={sm['comp']:.3f} Aff={sm['aff']:.3f} TD={sm['td']:.1f}")
        print("  Rankings -> Comp:", comp_rank, "| Aff:", aff_rank, "| TD:", td_rank)
        print("  Harmonic raw rank scores:", {k: round(v,4) for k,v in raw_rank_scores.items()})
        print("  Normalized weights:", {k: round(v,4) for k,v in weights_harmonic.items()})
        print(f"  => Best model (synthetic metrics aggregated): {best_model}")

        # 2) Chạy mô hình đã chọn trên dữ liệu THẬT, tính threshold, prediction
        chosen_raw = raw_scores[best_model]
        chosen_norm = normalize_scores(chosen_raw)
        threshold_best = compute_dynamic_threshold(chosen_norm, method=THRESH_METHOD, contamination=CONTAMINATION)
        y_pred = (chosen_norm > threshold_best).astype(int)
        length = len(y_true)
        y_true_pt = np.where(y_true > 0)[0]
        y_pred_pt = np.where(y_pred > 0)[0]

        # 3) Tính các metric trên NHÃN THẬT
        try:
            comp_f_score = Composite_f(length, y_true_pt, y_pred_pt).get_score()
        except Exception as e:
            if DEBUG_MODE:
                print(f"  [ERROR] Composite_f lỗi: {e}")
            comp_f_score = np.nan
        try:
            aff_f_score = Affiliation(length, y_true_pt, y_pred_pt).get_score()
        except Exception as e:
            if DEBUG_MODE:
                print(f"  [ERROR] Affiliation lỗi: {e}")
            aff_f_score = np.nan
        try:
            td_score = Temporal_distance(length, y_true_pt, y_pred_pt).get_score()
        except Exception as e:
            if DEBUG_MODE:
                print(f"  [ERROR] Temporal_distance lỗi: {e}")
            td_score = np.nan

        print(f"  Metrics on TRUE labels (best model {best_model}): CompF1={comp_f_score:.3f}, AffF1={aff_f_score:.3f}, TD={td_score:.1f}, Thr={threshold_best:.4f}, True={y_true.sum()}, Pred={y_pred.sum()}")

        evaluation_results.append({
            'placeId': place_id,
            'best_model': best_model,
            'comp_f1': comp_f_score,
            'aff_f1': aff_f_score,
            'td': td_score,
            'threshold': float(threshold_best),
            'num_true': int(y_true.sum()),
            'num_pred': int(y_pred.sum()),
            'len': length,
            'synth_comp': synth_metric_avg[best_model]['comp'],
            'synth_aff': synth_metric_avg[best_model]['aff'],
            'synth_td': synth_metric_avg[best_model]['td'],
        })

    print("\n" + "=" * 80)
    print("KẾT QUẢ CHỌN MÔ HÌNH BẰNG NHÃN GIẢ & ĐÁNH GIÁ TRÊN NHÃN THẬT")
    print("=" * 80)
    if not evaluation_results:
        print("Không có kết quả.")
    else:
        df_res = pd.DataFrame(evaluation_results)
        for _, r in df_res.iterrows():
            print(f"PlaceID {r.placeId}: Model={r.best_model} | CompF1={r.comp_f1:.3f} | AffF1={r.aff_f1:.3f} | TD={r.td:.1f} | True={r.num_true} Pred={r.num_pred} Len={r.len} Thr={r.threshold:.4f}")
        print("-" * 80)
        avg_comp_f1 = df_res.comp_f1.mean()
        avg_aff_f1 = df_res.aff_f1.mean()
        avg_td = df_res.td.mean()
        print(f"AVERAGE CompF1: {avg_comp_f1:.4f} trên {len(df_res)} placeId")
        print(f"AVERAGE AffF1: {avg_aff_f1:.4f}")
        print(f"AVERAGE TD: {avg_td:.4f} (càng thấp càng tốt)")

        out_csv = 'results_model_selection_synthetic_metrics.csv'
        df_res.to_csv(out_csv, index=False)
        summary_path = 'results_model_selection_synthetic_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('TÓM TẮT CHỌN MÔ HÌNH BẰNG NHÃN GIẢ\n')
            f.write('=' * 50 + '\n')
            f.write(f'Số placeId đã xử lý: {len(df_res)}\n\n')
            f.write(f'--- Composite F1 ---\n  Trung bình: {avg_comp_f1:.4f}\n  Cao nhất: {df_res.comp_f1.max():.4f}\n  Thấp nhất: {df_res.comp_f1.min():.4f}\n\n')
            f.write(f'--- Affiliation F1 ---\n  Trung bình: {avg_aff_f1:.4f}\n  Cao nhất: {df_res.aff_f1.max():.4f}\n  Thấp nhất: {df_res.aff_f1.min():.4f}\n\n')
            f.write(f'--- Temporal Distance ---\n  Trung bình: {avg_td:.4f}\n  Thấp nhất: {df_res.td.min():.4f}\n  Cao nhất: {df_res.td.max():.4f}\n')
            f.write('=' * 50 + '\n')
            f.write('TOP 5 THEO Composite F1:\n')
            top5 = df_res.sort_values('comp_f1', ascending=False).head(5)
            for _, row in top5.iterrows():
                f.write(f"PlaceID {row.placeId}: Model={row.best_model} CompF1={row.comp_f1:.4f} AffF1={row.aff_f1:.4f} TD={row.td:.1f}\n")
        print(f"\nĐã lưu kết quả chi tiết: {out_csv} và summary: {summary_path}")
# ==============================================================================
# test_and_visualize_single_id.py
#
# Script này thực hiện toàn bộ quy trình cho MỘT placeId duy nhất:
# 1. Đồng bộ dữ liệu và nhãn.
# 2. Lựa chọn mô hình tốt nhất bằng phương pháp không giám sát với VUS-ROC trên synthetic data.
# 3. Chạy mô hình đã chọn để tạo dự đoán.
# 4. Tính CompF1, AffF1, TD so với nhãn thật.
# 5. Trực quan hóa kết quả trên biểu đồ.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import anomaly_detection_base_model
import anomaly_injection_2 as anomaly_injection
import get_prediction_error_ranking

# Import metrics classes from main file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib, traceback
affiliation_pr = None
try:
    spec_metrics = importlib.util.find_spec('TSAD_eval.metrics')
    import TSAD_eval.metrics as _metrics_mod
    affiliation_pr = getattr(_metrics_mod, 'pr_from_events', None)
except Exception as e_pkg:
    TSAD_DIR = os.path.join(PROJECT_ROOT, 'TSAD_eval')
    if TSAD_DIR not in sys.path:
        sys.path.insert(0, TSAD_DIR)
    try:
        import TSAD_eval.metrics as _metrics_mod
        affiliation_pr = getattr(_metrics_mod, 'pr_from_events', None)
    except Exception as e_fb:
        print(f"[WARN] Cannot import TSAD metrics, using fallback: {e_fb}")
        affiliation_pr = None

# Copy metric classes from main script
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

# VUS-ROC functions
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

# Helper functions (VUS-ROC and model selection logic)
def normalize_scores(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def dynamic_threshold(scores, method='z', contamination=0.05):
    scores = np.asarray(scores)
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
        raise ValueError('Unsupported threshold method')

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

warnings.filterwarnings('ignore')

# ==============================================================================
# PHẦN B: KỊCH BẢN CHÍNH - SỬ DỤNG VUS-ROC TRÊN SYNTHETIC DATA ĐỂ CHỌN MODEL
# ==============================================================================
if __name__ == '__main__':
    # --- 1. CẤU HÌNH ---
    DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'labels'
    NUM_PLACE_IDS = 30  # Số placeId đầu tiên sẽ chạy

    print(f"--- BẮT ĐẦU XỬ LÝ {NUM_PLACE_IDS} placeId ĐẦU TIÊN VỚI TWO-JUDGE SELECTION ---")

    # Đọc dữ liệu nguồn một lần
    data_full = pd.read_csv(DATA_FILE_PATH)
    if 'placeId' not in data_full.columns:
        raise RuntimeError("Cột 'placeId' không tồn tại trong dữ liệu.")
    data_full['placeId_norm'] = pd.to_numeric(data_full['placeId'], errors='coerce').astype('Int64')

    unique_place_ids = data_full['placeId_norm'].dropna().unique().tolist()
    selected_place_ids = unique_place_ids[:NUM_PLACE_IDS]
    if not selected_place_ids:
        raise SystemExit("Không tìm thấy placeId nào trong dữ liệu.")

    # Danh sách hàm điểm số (đồng bộ với main script)
    model_funcs = {
        'SR': anomaly_detection_base_model.run_sr_scores,
        'IQR': anomaly_detection_base_model.run_iqr_scores,
        'MA': anomaly_detection_base_model.run_moving_average_scores,
        'KNN': anomaly_detection_base_model.run_knn_scores,
        'RePAD': anomaly_detection_base_model.run_repad_scores,
        'Prophet': anomaly_detection_base_model.run_prophet_scores,
        'Moment': anomaly_detection_base_model.run_moment_scores
    }
    model_names = list(model_funcs.keys())

    images_dir = 'images_after_Prophet_3'
    os.makedirs(images_dir, exist_ok=True)

    summary_rows = []

    for idx, place_id in enumerate(selected_place_ids, start=1):
        print(f"\n================= ({idx}/{len(selected_place_ids)}) PlaceID: {place_id} =================")
        ts_source = data_full[data_full['placeId_norm'] == place_id].copy()
        if ts_source.empty:
            print("  -> Bỏ qua: không có dữ liệu.")
            continue

        label_path = os.path.join(LABELS_DIR, f"label_{place_id}.csv")
        if not os.path.exists(label_path):
            alt_label_path = os.path.join('..', 'labels', f'label_{place_id}.csv')
            if os.path.exists(alt_label_path):
                label_path = alt_label_path
            else:
                print("  -> Bỏ qua: không tìm thấy file nhãn.")
                continue
        df_label = pd.read_csv(label_path)

        # Chuẩn hoá kiểu dữ liệu ngày
        ts_source['date'] = ts_source['date'].astype(str)
        df_label['date'] = df_label['date'].astype(str)

        merged_df = pd.merge(ts_source, df_label, on='date', how='inner', suffixes=('_main', '_label'))
        if merged_df.empty:
            print("  -> Bỏ qua: không có intersection ngày giữa dữ liệu và nhãn.")
            continue

        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date')
        dates = merged_df['date']
        views = merged_df['view_main']
        y_true = (merged_df['label'].fillna(0).astype(float) > 0).astype(int).to_numpy()

        length = len(y_true)
        y_true_pt = np.where(y_true > 0)[0]
        ts_np = views.to_numpy()
        series_df = pd.DataFrame({'view': views})

        # --- GIAI ĐOẠN 1: CHỌN MÔ HÌNH BẰNG TWO-JUDGE SYSTEM ---
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
        
        # Judge 1: VUS-ROC trên synthetic data
        metrics_per_test = {}
        for test_name, inject_func in injection_tests.items():
            injected_series, pseudo_labels = inject_func(ts_np)
            if len(np.unique(pseudo_labels)) < 2:
                print(f"  [INFO] Bỏ qua test {test_name} vì pseudo_labels chỉ có 1 lớp")
                continue
            
            length_inj = len(pseudo_labels)
            test_metrics = {}
            injected_df = pd.DataFrame({'view': injected_series})
            
            for m_name, fn in model_funcs.items():
                try:
                    s_inj = fn(injected_df)
                except Exception as e_model_inj:
                    print(f"    [WARN] Model {m_name} lỗi trên test {test_name}: {e_model_inj}")
                    s_inj = np.zeros(length_inj)
                
                s_norm = normalize_scores(np.asarray(s_inj))
                vus_score_inj = compute_vus_roc(s_norm, pseudo_labels, max_ell=length_inj//10)
                test_metrics[m_name] = vus_score_inj
            metrics_per_test[test_name] = test_metrics

        # In VUS-ROC cho từng loại injection và model
        print("  Judge 1 - VUS-ROC scores per injection test:")
        for test_name in metrics_per_test:
            print(f"    {test_name}:", {m: f"{metrics_per_test[test_name].get(m, 0):.3f}" for m in model_names})

        # Gộp VUS-ROC trung bình qua các test cho mỗi model
        synth_metric_avg = {m: np.nanmean([metrics_per_test.get(t, {}).get(m, np.nan) for t in metrics_per_test]) for m in model_names}
        
        valid_models_vus = [m for m in model_names if not np.isnan(synth_metric_avg[m])]
        vus_ranking = sorted(valid_models_vus, key=lambda m: synth_metric_avg[m], reverse=True)
        
        print(f"  Judge 1 - VUS-ROC averages: {dict((k, f'{v:.3f}') for k, v in synth_metric_avg.items())}")
        print(f"  Judge 1 - VUS-ROC ranking: {vus_ranking}")
        
        # Judge 2: Prediction Error correlation ranking
        pred_error_ranking, pred_error_correlations = get_prediction_error_ranking.get_prediction_error_ranking(ts_np, model_funcs)
        
        print(f"  Judge 2 - Prediction Error ranking: {pred_error_ranking}")
        print(f"  Judge 2 - Error correlations: {dict((k, f'{v:.3f}') for k, v in pred_error_correlations.items())}")
        
        # TÍNH TRỌNG SỐ TỪ COMBO 2 RANKING (HARMONIC AGGREGATION)
        ranking_lists = [vus_ranking, pred_error_ranking]
        weights, raw_scores = derive_weights_from_rankings(ranking_lists, model_names)
        
        print(f"  Combined harmonic scores: {dict((k, f'{v:.3f}') for k, v in raw_scores.items())}")
        print(f"  Two-Judge ensemble weights: {dict((k, f'{v:.3f}') for k, v in weights.items())}")
        
        # Chọn best model dựa trên trọng số cao nhất
        best_model = max(weights, key=weights.get)
        
        print(f"  => Best model (Two-Judge): {best_model}")

        # --- GIAI ĐOẠN 2: CHẠY BEST MODEL TRÊN DỮ LIỆU THẬT VÀ ĐÁNH GIÁ ---
        try:
            best_scores = model_funcs[best_model](series_df)
        except Exception as e:
            print(f"  [ERROR] Best model {best_model} lỗi: {e}")
            best_scores = np.zeros(length)
        
        best_norm = normalize_scores(np.asarray(best_scores))
        threshold = dynamic_threshold(best_norm, method='z')
        y_pred = (best_norm > threshold).astype(int)
        y_pred_pt = np.where(y_pred > 0)[0]

        # Tính metrics trên nhãn thật
        try:
            comp_f1 = Composite_f(length, y_true_pt, y_pred_pt).get_score()
        except Exception as e:
            print(f"  [WARN] Composite_f lỗi: {e}")
            comp_f1 = np.nan
        try:
            aff_f1 = Affiliation(length, y_true_pt, y_pred_pt).get_score()
        except Exception as e:
            print(f"  [WARN] Affiliation lỗi: {e}")
            aff_f1 = np.nan
        try:
            td_val = Temporal_distance(length, y_true_pt, y_pred_pt).get_score()
        except Exception as e:
            print(f"  [WARN] Temporal_distance lỗi: {e}")
            td_val = np.nan

        print(f"  (Best) {best_model} | CompF1={comp_f1:.4f} | AffF1={aff_f1:.4f} | TD={td_val:.1f} | Thr={threshold:.4f} | True={y_true.sum()} | Pred={y_pred.sum()}")

        # Plot visualization
        true_anomaly_dates = dates[y_true == 1]
        true_anomaly_views = views[y_true == 1]
        predicted_anomaly_dates = dates[y_pred == 1]
        predicted_anomaly_views = views[y_pred == 1]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(dates, views, label='Views', zorder=1)
        ax.scatter(true_anomaly_dates, true_anomaly_views, s=100, color='red', marker='o', edgecolor='black', label='GT', zorder=2)
        ax.scatter(predicted_anomaly_dates, predicted_anomaly_views, s=150, facecolors='none', marker='o', edgecolor='limegreen', linewidth=2, label='Pred', zorder=3)
        ax.set_title(
            f"PlaceID {place_id} | Best={best_model} (Two-Judge) | CompF1={comp_f1:.3f} | AffF1={aff_f1:.3f} | TD={td_val:.0f}"
        )
        ax.legend(fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(images_dir, f"visualization_{place_id}.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  -> Đã lưu biểu đồ: {out_path}")

        summary_rows.append({
            'placeId': place_id,
            'num_points': len(merged_df),
            'num_true_anomalies': int(y_true.sum()),
            'num_pred_anomalies': int(y_pred.sum()),
            'best_model': best_model,
            'composite_f1': comp_f1,
            'affiliation_f1': aff_f1,
            'temporal_distance': td_val,
            'threshold': threshold,
            'vus_roc_avg': synth_metric_avg[best_model],
            'pred_error_top': pred_error_ranking[0] if pred_error_ranking else 'N/A',
            'two_judge_weight': weights[best_model]
        })

    # Ghi summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values('composite_f1', ascending=False)
        summary_path = 'ensemble_two_judge_test_30_placeIds_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n==> ĐÃ LƯU SUMMARY: {summary_path}")
        print(summary_df.head())
    else:
        print("\nKhông có placeId nào được xử lý thành công.")
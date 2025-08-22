import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import sys

# --- PHẦN 0: IMPORTS VÀ CÀI ĐẶT ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import VUS-ROC từ timeeval
try:
    from timeeval.metrics.vus_metrics import RangeRocVUS
except ImportError:
    print("LỖI: Cần thư viện 'timeeval'. Chạy: pip install timeeval")
    exit()

# Import các mô hình và injector
try:
    import anomaly_detection_base_model as ad_models
    from anomaly_injection_2 import AnomalyInjector
except ImportError:
    print("LỖI: Không tìm thấy 'anomaly_detection_base_model.py' hoặc 'anomaly_injection_2.py'.")
    exit()

# Import các metric CompF1, AffF1, TD từ TSAD_eval
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

# --- METRIC CLASSES IMPLEMENTATION ---
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


# --- PHẦN 1: CÁC HÀM TIỆN ÍCH ---

def normalize_scores(scores):
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val - min_val > 1e-6:
        return (scores - min_val) / (max_val - min_val)
    return np.zeros_like(scores)

def compute_dynamic_threshold(scores: np.ndarray, contamination: float = 0.03) -> float:
    if scores.size == 0: return np.inf
    return np.percentile(scores, 100 * (1 - contamination))

def visualize_ensemble_result(series, true_labels, predicted_labels, place_id, output_dir, test_index):
    plt.figure(figsize=(20, 8))
    plt.plot(series, label='Original Time Series', color='royalblue', alpha=0.8, zorder=2)
    
    true_indices = np.where(true_labels == 1)[0]
    if len(true_indices) > 0:
        for i, idx in enumerate(true_indices):
            plt.axvspan(idx - 0.5, idx + 0.5, color='gold', alpha=0.6, zorder=1, 
                        label='Ground Truth Anomaly' if i == 0 else "")

    predicted_indices = np.where(predicted_labels == 1)[0]
    if len(predicted_indices) > 0:
        plt.scatter(predicted_indices, series[predicted_indices], 
                    color='red', marker='o', s=60, zorder=3, 
                    label='Ensemble Prediction')

    plt.title(f'Test {test_index}', fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    file_name = f'test_{test_index}.png'
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path)
    plt.close()

# --- PHẦN 2: LOGIC CHÍNH ---

DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
LABEL_DIR = 'labels'
OUTPUT_DIR = 'ensemble_results_final'

def main():
    print("Bắt đầu quy trình Weighted Ensemble cho từng PlaceID...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_full = pd.read_csv(DATA_FILE_PATH)
    all_place_ids = data_full['placeId'].unique()[:30]
    
    models_to_evaluate = {
        'SR': ad_models.run_sr_scores, 
        'IQR': ad_models.run_iqr_scores, 
        'MA': ad_models.run_moving_average_scores,
        # 'IForest': ad_models.run_iforest_scores, 
        'KNN': ad_models.run_knn_scores, 
        'RePAD': ad_models.run_repad_scores,
        # 'Prophet': ad_models.run_prophet_scores, 
        'Moment': ad_models.run_moment_scores
    }

    final_metrics_summary = []

    # --- Vòng lặp chính: Xử lý từng PlaceID một cách độc lập ---
    for test_index, place_id in enumerate(tqdm(all_place_ids, desc="Processing PlaceIDs"), start=1):
        print(f"\n===== Test {test_index}: Đang xử lý PlaceID: {place_id} =====")
        original_series = data_full[data_full['placeId'] == place_id].sort_values('date')['view'].to_numpy().astype(float)
        n = len(original_series)
        if n < 100:
            print(f"Bỏ qua Test {test_index} (PlaceID {place_id}) do chuỗi quá ngắn (dài {n} < 100).")
            continue

        # --- BƯỚC 1: TÌM TRỌNG SỐ CHO PLACEID HIỆN TẠI ---
        print("  BƯỚC 1: Đánh giá mô hình trên nhãn giả để tìm trọng số...")
        injector = AnomalyInjector(random_state=42)
        
        profile = {
            "global": {"n_anomalies": 3, "magnitude_std": 7.0},
            # "contextual": {"n_anomalies": 3, "magnitude_std": 5.0},
            "trend": {"min_len": n//15, "max_len": n//8, "magnitude": np.std(original_series) * 2.0},
            "seasonal": {"n_anomalies": 4, "magnitude_std": 6.0, "period": 7},
            "cutoff": {"min_len": n//20, "max_len": n//10},
            "average": {"min_len": n//15, "max_len": n//8}
        }
        
        vus_roc_calculator = RangeRocVUS(max_buffer_size=n // 10, compatibility_mode=True)
        evaluation_results = {model_name: [] for model_name in models_to_evaluate}

        for anomaly_type, params in profile.items():
            injected_series, pseudo_labels = injector.inject(original_series, anomaly_type, params)
            injected_df = pd.DataFrame(injected_series)
            if np.sum(pseudo_labels) == 0: continue

            for model_name, model_func in models_to_evaluate.items():
                try:
                    anomaly_scores = model_func(injected_df)
                    score = vus_roc_calculator(y_true=pseudo_labels.astype(np.float64), y_score=np.array(anomaly_scores).astype(np.float64))
                    evaluation_results[model_name].append(score)
                except Exception:
                    evaluation_results[model_name].append(0.0)
        
        avg_vus_scores = {model: np.mean(scores) if scores else 0 for model, scores in evaluation_results.items()}
        total_vus_score = sum(avg_vus_scores.values())
        
        if total_vus_score < 1e-9:
            weights = {model: 1.0 / len(models_to_evaluate) for model in models_to_evaluate}
        else:
            weights = {model: score / total_vus_score for model, score in avg_vus_scores.items()}
        
        print("  Trọng số đã tính cho PlaceID này:")
        for model, weight in weights.items(): print(f"    - {model:<10}: {weight:.4f}")

        # --- BƯỚC 2: DỰ ĐOÁN TRÊN DỮ LIỆU THẬT VÀ ĐÁNH GIÁ ---
        print("  BƯỚC 2: Áp dụng Ensemble và đánh giá trên nhãn thật...")
        
        label_path = os.path.join(LABEL_DIR, f'label_{place_id}.csv')
        try:
            label_df = pd.read_csv(label_path)
            true_labels = label_df['label'].to_numpy()[:n]
        except FileNotFoundError:
            true_labels = np.zeros(n, dtype=int)
            
        ensemble_scores = np.zeros(n, dtype=float)
        series_df = pd.DataFrame(original_series)

        for model_name, model_func in models_to_evaluate.items():
            try:
                scores = model_func(series_df)
                ensemble_scores += normalize_scores(np.array(scores)) * weights[model_name]
            except Exception:
                continue
                
        threshold = compute_dynamic_threshold(ensemble_scores)
        predicted_labels = (ensemble_scores > threshold).astype(int)
        
        # Đánh giá bằng các metric từ TSAD_eval
        y_true_pt = np.where(true_labels > 0)[0]
        y_pred_pt = np.where(predicted_labels > 0)[0]
        
        comp_f1 = Composite_f(n, y_true_pt, y_pred_pt).get_score()
        aff_f1 = Affiliation(n, y_true_pt, y_pred_pt).get_score()
        td = Temporal_distance(n, y_true_pt, y_pred_pt).get_score()
        
        final_metrics_summary.append({
            'placeId': place_id, 'CompF1': comp_f1, 'AffF1': aff_f1, 'TD': td
        })

        visualize_ensemble_result(original_series, true_labels, predicted_labels, place_id, OUTPUT_DIR, test_index)

    # --- BƯỚC 3: BÁO CÁO KẾT QUẢ TỔNG HỢP ---
    if final_metrics_summary:
        results_df = pd.DataFrame(final_metrics_summary)
        print("\n" + "="*80)
        print("BƯỚC 3: KẾT QUẢ CUỐI CÙNG TRÊN TẤT CẢ CÁC PLACEID")
        print("="*80)
        print(results_df.round(3).to_string(index=False))
        print("-" * 80)
        print("Trung bình toàn cục:")
        print(results_df.drop('placeId', axis=1).mean())
        print("="*80)
        
        output_file = os.path.join(OUTPUT_DIR, 'final_evaluation_metrics_per_placeid.csv')
        results_df.to_csv(output_file, index=False)
        print(f"Đã lưu kết quả metric vào: {output_file}")

    print(f"\nHoàn thành! Tất cả đồ thị đã được lưu vào thư mục: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
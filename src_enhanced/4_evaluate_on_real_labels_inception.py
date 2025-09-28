# evaluate_on_real_labels_inception.py

import os
import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
import warnings
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

# --- PHẦN 0: CÀI ĐẶT VÀ IMPORT ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# <<< THAY ĐỔI 1: Thêm import cho pycatch22 và tsfresh >>>
try:
    from pycatch22 import catch22_all
    from tsfresh import extract_features
    from tsfresh.feature_extraction import ComprehensiveFCParameters
    from sklearn.preprocessing import MinMaxScaler
    # <<< THAY ĐỔI 2: Xóa import cũ không cần thiết >>>
    # from extract_full_features import extract_full_meta_features 
except ImportError as e:
    print(f"LỖI: Thiếu thư viện hoặc module. Vui lòng kiểm tra lại. Lỗi: {e}")
    exit()

# --- METRIC CLASSES IMPLEMENTATION (Giữ nguyên không đổi) ---
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
# --- HẾT PHẦN METRIC CLASSES ---

# --- PHẦN 1: CẤU HÌNH ---

DATA_FILE = '/content/drive/MyDrive/LSTM-VAE/data/cleaned_data_no_zero_periods_filtered.csv'
LABEL_DIR = '/content/drive/MyDrive/LSTM-VAE/labels'
META_MODEL_FILE = '/content/drive/MyDrive/InceptionTime/inception_model_output/best_model.h5'
METADATA_FILE = '/content/drive/MyDrive/InceptionTime/results_enhanced_2/meta_model_metadata.joblib' 
OUTPUT_DIR = '/content/drive/MyDrive/InceptionTime/results_evaluation_inception/'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- PHẦN 2: CÁC HÀM TIỆN ÍCH ---

# <<< THAY ĐỔI 3: Thêm hàm trích xuất đặc trưng kết hợp từ file generate_meta_feature_matrix.py >>>
def extract_combined_features(series_data, series_id):
    """
    Trích xuất đặc trưng từ cả Catch22 và tsfresh.
    series_data là một DataFrame có cột 'date' và 'view'.
    """
    # 1. Trích xuất từ Catch22
    catch22_features = catch22_all(series_data['view'].values)
    feature_row = {name: val for name, val in zip(catch22_features['names'], catch22_features['values'])}

    # 2. Chuẩn bị dữ liệu cho tsfresh
    df_tsfresh = series_data[['date', 'view']].copy()
    df_tsfresh = df_tsfresh.rename(columns={'date': 'time', 'view': 'value'})
    df_tsfresh['id'] = series_id

    # 3. Trích xuất từ tsfresh
    settings = ComprehensiveFCParameters()
    tsfresh_features = extract_features(df_tsfresh, column_id='id', column_sort='time',
                                        default_fc_parameters=settings,
                                        disable_progressbar=True)

    # 4. Kết hợp hai bộ đặc trưng
    if not tsfresh_features.empty:
        tsfresh_dict = {f"tsf_{key}": val for key, val in tsfresh_features.iloc[0].to_dict().items()}
        feature_row.update(tsfresh_dict)

    return feature_row

def load_data(file_path):
    """Tải dữ liệu chuỗi thời gian từ file CSV."""
    print(f"\nĐang tải dữ liệu từ '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file dữ liệu tại '{file_path}'")
        return None
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values(by=['placeId', 'date'])
    all_series = {pid: group.set_index('date')['view'] for pid, group in df.groupby('placeId')}
    print(f"Tải thành công {len(all_series)} chuỗi thời gian.")
    return all_series

def load_true_labels(place_id):
    """Tải nhãn thật từ file CSV tương ứng."""
    label_file = os.path.join(LABEL_DIR, f"label_{place_id}.csv")
    if not os.path.exists(label_file):
        return None
    df = pd.read_csv(label_file)
    return df['label'].values

def visualize_result(series, true_labels, predicted_labels, place_id, scores, test_index):
    """Trực quan hóa kết quả so sánh."""
    plt.figure(figsize=(20, 8))
    plt.plot(series.values, label='Original Time Series', color='royalblue', alpha=0.8, zorder=2)
    
    true_indices = np.where(true_labels == 1)[0]
    if len(true_indices) > 0:
        for i, idx in enumerate(true_indices):
            plt.axvspan(idx - 0.5, idx + 0.5, color='gold', alpha=0.6, zorder=1, 
                        label='Ground Truth Anomaly' if i == 0 else "")

    predicted_indices = np.where(predicted_labels == 1)[0]
    if len(predicted_indices) > 0:
        plt.scatter(predicted_indices, series.values[predicted_indices], 
                    color='red', marker='o', s=60, zorder=3, 
                    label='Ensemble Prediction')

    plt.title(f'Test {test_index} (PlaceID: {place_id}) - CompF1: {scores["CompF1"]:.2f}', fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    file_name = f'test_{test_index}_{place_id}.png'
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path)
    plt.close()


# --- PHẦN 3: HÀM THỰC THI CHÍNH ---

def main():
    print("="*80)
    print("Bắt đầu Giai đoạn 2: Đánh giá trên Nhãn thật (sử dụng Meta-Model InceptionTime)")
    print("="*80)

    # 1. Tải Meta-Model và metadata
    print(f"Đang tải Meta-Model từ '{META_MODEL_FILE}'...")
    if not os.path.exists(META_MODEL_FILE) or not os.path.exists(METADATA_FILE):
        print("LỖI: Không tìm thấy file model hoặc metadata.")
        return
    
    # Load model và metadata
    meta_model = tf.keras.models.load_model(META_MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)
    model_names = metadata['performance_model_names']
    scaler_X = metadata['scaler_X'] # Tải scaler cho features
    scaler_y = metadata['scaler_y'] # Tải scaler cho target
    print("✓ Đã tải thành công meta-model và metadata (bao gồm cả scalers).")

    # (Giữ nguyên phần tạo dictionary các hàm mô hình cơ sở)
    available_models = {
        "SR": run_sr_scores,
        "IQR": run_iqr_scores,
        "KNN": run_knn_scores,
        "MA": run_moving_average_scores,
        "MOMENT": run_moment_scores,
        'RePAD': run_repad_scores,
    }
    base_models_to_run = {name: func for name, func in available_models.items() if name in model_names}

    # 2. Tải dữ liệu và nhãn
    all_series = load_data(DATA_FILE)
    if not all_series:
        return

    all_series_items = list(all_series.items())[:30]
    print(f"Giới hạn xử lý: {len(all_series_items)} địa điểm đầu tiên.")

    evaluation_results = []
    weights_results = []

    # 3. Lặp qua từng chuỗi thời gian để đánh giá
    print(f"\nBắt đầu đánh giá trên {len(all_series_items)} địa điểm...")
    for test_index, (place_id, series) in enumerate(tqdm(all_series_items, desc="Đang đánh giá"), start=1):
        
        # <<< THAY ĐỔI 4: Cập nhật logic trích xuất meta-features >>>
        # 3.1. Trích xuất meta-features kết hợp (Catch22 + tsfresh)
        print(f"\nĐang trích xuất meta-features cho place_id {place_id}...")
        
        # Chuyển đổi pandas Series thành DataFrame mà hàm extract_combined_features yêu cầu
        series_df = series.reset_index()
        series_df.columns = ['date', 'view'] # Đảm bảo tên cột chính xác
        
        # Gọi hàm mới để trích xuất
        all_features = extract_combined_features(series_df, place_id)
        
        # Sắp xếp features theo thứ tự giống như trong training
        feature_names = metadata['meta_feature_names']
        meta_features_values = []
        for feature_name in feature_names:
            if feature_name in all_features:
                meta_features_values.append(all_features[feature_name])
            else:
                # Nếu một feature nào đó không được trích xuất (ví dụ do lỗi), gán giá trị 0
                meta_features_values.append(0.0)
        
        meta_features_flat = np.array(meta_features_values).reshape(1, -1)
        
        # Kiểm tra và xử lý NaN/Inf trong meta-features
        meta_features_flat = np.nan_to_num(meta_features_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Chuẩn hóa meta-features bằng scaler_X đã được train
        meta_features_normalized = scaler_X.transform(meta_features_flat)

        # Reshape để phù hợp với đầu vào của InceptionTime: (1, num_features, 1)
        meta_features_reshaped = meta_features_normalized.reshape((1, meta_features_normalized.shape[1], 1))

        # 3.2. Dự đoán hiệu suất (trọng số) bằng Meta-Model InceptionTime
        predicted_weights_normalized = meta_model.predict(meta_features_reshaped, verbose=0)[0]
        
        # Denormalize predictions về không gian ban đầu bằng scaler_y
        predicted_weights_raw = scaler_y.inverse_transform(predicted_weights_normalized.reshape(1, -1))[0]
        
        # Chuẩn hóa trọng số bằng softmax để đảm bảo tổng = 1
        predicted_weights_exp = np.exp(predicted_weights_raw - np.max(predicted_weights_raw))
        predicted_weights = predicted_weights_exp / np.sum(predicted_weights_exp)
        
        weights_dict = dict(zip(model_names, predicted_weights))
        weights_results.append({'placeId': place_id, 'test_index': test_index, **weights_dict})

        # 3.3. Chạy các mô hình cơ sở để lấy điểm
        all_scores = {}
        for model_name, model_func in base_models_to_run.items():
            try:
                scores = model_func(pd.DataFrame({'value': series.values}))
                # Chuẩn hóa điểm của từng model về [0, 1]
                min_score, max_score = np.min(scores), np.max(scores)
                if max_score > min_score:
                    all_scores[model_name] = (scores - min_score) / (max_score - min_score)
                else:
                    all_scores[model_name] = np.zeros(len(series))
            except Exception as e:
                print(f"Lỗi khi chạy model {model_name} cho place_id {place_id}: {e}")
                all_scores[model_name] = np.zeros(len(series))

        # 3.4. Tính điểm tổ hợp có trọng số (Weighted Ensemble Score)
        ensemble_scores = np.zeros(len(series))
        for model_name, scores in all_scores.items():
            ensemble_scores += scores * weights_dict.get(model_name, 0)

        # 3.5. Tải nhãn thật
        true_labels = load_true_labels(place_id)
        if true_labels is None or len(true_labels) != len(series):
            print(f"Bỏ qua place_id {place_id} do không có nhãn hoặc độ dài không khớp.")
            continue
        
        # 3.6. Chuyển điểm thành nhãn dự đoán
        threshold = np.percentile(ensemble_scores, 99)
        predicted_labels = (ensemble_scores > threshold).astype(int)

        # 3.7. Đánh giá bằng các metric
        n = len(series)
        y_true_indices = np.where(true_labels > 0)[0]
        y_pred_indices = np.where(predicted_labels > 0)[0]
        
        if len(y_true_indices) > 0:
            comp_f1 = Composite_f(n, y_true_indices, y_pred_indices).get_score()
            aff_f1 = Affiliation(n, y_true_indices, y_pred_indices).get_score()
            td = Temporal_distance(n, y_true_indices, y_pred_indices).get_score()
        else:
            # Nếu không có bất thường thật, điểm F1 không xác định, có thể gán là 0
            comp_f1, aff_f1, td = 0.0, 0.0, float('inf')

        scores = {'CompF1': comp_f1, 'AffF1': aff_f1, 'TD': td}
        evaluation_results.append({'placeId': place_id, **scores})

        # 3.8. Trực quan hóa và lưu kết quả
        visualize_result(series, true_labels, predicted_labels, place_id, scores, test_index)
    
    # 4. In và lưu kết quả tổng hợp
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results).set_index('placeId')
        
        print("\n" + "="*80)
        print("KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP (INCEPTIONTIME)")
        print("="*80)
        print(results_df)
        
        avg_scores = results_df.mean()
        avg_scores.name = 'AVERAGE'
        results_df = pd.concat([results_df, avg_scores.to_frame().T])
        
        print("\nĐiểm trung bình trên tất cả các địa điểm:")
        print(results_df.loc['AVERAGE'])
        
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_summary.csv'))
        print(f"\nKết quả tổng hợp đã được lưu vào: '{os.path.join(OUTPUT_DIR, 'evaluation_summary.csv')}'")

    # 5. Lưu và hiển thị file trọng số
    if weights_results:
        weights_df = pd.DataFrame(weights_results)
        weights_output_file = os.path.join(OUTPUT_DIR, 'model_weights_per_location.csv')
        weights_df.to_csv(weights_output_file, index=False)
        print(f"\nTrọng số các mô hình đã được lưu vào: '{weights_output_file}'")

if __name__ == '__main__':
    main()
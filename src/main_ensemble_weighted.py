import os
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

# Tắt TensorFlow / các cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import anomaly_detection_base_model
import anomaly_injection
import model_centrality
import get_prediction_error_ranking
import borda_count_rank_aggregation


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
    """
    Tạo trọng số cho từng model dựa trên nhiều bảng xếp hạng.
    Chiến lược:
      - Với mỗi bảng xếp hạng: model ở vị trí i nhận điểm = 1 / (i+1)
      - Tổng hợp điểm qua tất cả bảng → weight thô
      - Chuẩn hóa về tổng = 1
    Nếu tất cả weight = 0 (bất thường) → fallback weight đều nhau.
    """
    raw_scores = {m: 0.0 for m in models_order}
    for ranking in ranking_lists:
        for i, m in enumerate(ranking):
            if m in raw_scores:
                raw_scores[m] += 1.0 / (i + 1.0)
    total = sum(raw_scores.values())
    if total <= 0:
        # Fallback: đều nhau
        equal = 1.0 / len(models_order)
        return {m: equal for m in models_order}, raw_scores
    weights = {m: v / total for m, v in raw_scores.items()}
    return weights, raw_scores


def derive_weights_from_borda(borda_scores: dict):
    """Chuyển điểm Borda thành trọng số chuẩn hóa."""
    total = sum(borda_scores.values())
    if total <= 0:
        n = len(borda_scores)
        return {m: 1.0 / n for m in borda_scores}
    return {m: s / total for m, s in borda_scores.items()}


def weighted_aggregate(scores_dict: dict, weights: dict) -> np.ndarray:
    """
    Tính tổng hợp có trọng số từ dict model -> normalized scores.
    Chỉ lấy giao các model có cả scores và weight.
    """
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
    if total_w <= 1e-12:  # Fallback chia đều
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
            return mean_ + 1e6  # Không tạo anomaly
        return mean_ + 2.5 * std_
    elif method == 'percentile':
        return np.percentile(scores, 100 * (1 - contamination))
    else:
        raise ValueError("method phải là 'z' hoặc 'percentile'")


if __name__ == '__main__':
    # --- Tham số chính ---
    DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'labels/'
    MAX_PLACEIDS = 30
    THRESH_METHOD = 'z'          # 'z' hoặc 'percentile'
    CONTAMINATION = 0.05         # Dùng nếu method='percentile'
    WEIGHT_STRATEGY = 'harmonic'    # 'borda' hoặc 'harmonic'

    print("Đang đọc dữ liệu...")
    try:
        data_full = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        raise SystemExit(f"Không tìm thấy file dữ liệu: {DATA_FILE_PATH}")

    unique_place_ids = data_full['placeId'].unique()
    place_ids_to_process = unique_place_ids[:MAX_PLACEIDS]
    data_subset = data_full[data_full['placeId'].isin(place_ids_to_process)]

    print("\nBắt đầu Weighted Ensemble dựa trên giám khảo...")
    evaluation_results = []

    model_funcs = {
        'SR': anomaly_detection_base_model.run_sr_scores,
        'IQR': anomaly_detection_base_model.run_iqr_scores,
        'MA': anomaly_detection_base_model.run_moving_average_scores,
        'IForest': anomaly_detection_base_model.run_iforest_scores,
        'KNN': anomaly_detection_base_model.run_knn_scores,
        'RePAD': anomaly_detection_base_model.run_repad_scores, 
        'Prophet': anomaly_detection_base_model.run_prophet_scores,
        # 'Windowed_IForest': anomaly_detection_base_model.run_windowed_iforest_scores
    }
    model_names = list(model_funcs.keys())

    for place_id in tqdm(place_ids_to_process, desc="Processing PlaceIDs"):
        ts_group = data_subset[data_subset['placeId'] == place_id].sort_values('date')
        series_df = ts_group[['view']]

        label_path = os.path.join(LABELS_DIR, f'label_{place_id}.csv')
        if not os.path.exists(label_path):
            print(f"[Skip] Không có label cho {place_id}")
            continue
        df_label = pd.read_csv(label_path)
        y_true = df_label['label'].to_numpy()
        if len(series_df) != len(y_true):
            print(f"[Skip] Length mismatch {place_id}")
            continue

        # Lấy raw scores
        raw_scores = {}
        for m, fn in model_funcs.items():
            try:
                s = fn(series_df)
            except Exception as e:
                print(f"Model {m} error {e}; dùng zeros")
                s = np.zeros(len(series_df))
            raw_scores[m] = s

        ts_np = series_df['view'].to_numpy()
        print(f"\n>>> PlaceID {place_id}: Running judges")
        syn_rank, syn_scores = anomaly_injection.get_synthetic_ranking(ts_np, model_funcs)
        # cen_rank, cen_scores = model_centrality.get_centrality_ranking(ts_np, model_funcs)
        pe_rank, pe_scores = get_prediction_error_ranking.get_prediction_error_ranking(ts_np, model_funcs)

        all_rankings = [syn_rank, pe_rank]
        final_ranking, borda_scores = borda_count_rank_aggregation.borda_count_aggregation(all_rankings)

        # Hiển thị ranking từ từng "giám khảo"
        print("  Synthetic ranking (Giám khảo 1):", syn_rank)
        # print("  Centrality ranking (Giám khảo 2):", cen_rank)
        print("  Prediction-Error ranking (Giám khảo 3):", pe_rank)

        if WEIGHT_STRATEGY == 'borda':
            weights = derive_weights_from_borda(borda_scores)
        else:
            weights, _ = derive_weights_from_rankings(all_rankings, model_names)

        normalized_scores = {m: normalize_scores(raw_scores[m]) for m in model_names}
        aggregated_scores = weighted_aggregate(normalized_scores, weights)

        threshold = compute_dynamic_threshold(aggregated_scores, method=THRESH_METHOD, contamination=CONTAMINATION)
        y_pred = (aggregated_scores > threshold).astype(int)

        f1 = f1_score(y_true, y_pred)
        evaluation_results.append({
            'placeId': place_id,
            'f1_score': f1,
            'num_true': int(np.sum(y_true)),
            'num_pred': int(np.sum(y_pred)),
            'len': len(y_true),
            'threshold': float(threshold)
        })

    print(f"PlaceID {place_id} -> F1={f1:.3f} | True={np.sum(y_true)} Pred={np.sum(y_pred)}")
    print("  Final aggregated ranking:", final_ranking)
    print("  Weights:", {k: round(v, 3) for k, v in weights.items()})

    print("\n" + "=" * 60)
    print("KẾT QUẢ WEIGHTED ENSEMBLE")
    print("=" * 60)
    if not evaluation_results:
        print("Không có kết quả.")
    else:
        df_res = pd.DataFrame(evaluation_results)
        for _, r in df_res.iterrows():
            print(f"PlaceID {r.placeId}: F1={r.f1_score:.3f} (True={r.num_true}, Pred={r.num_pred}, Len={r.len}, Thr={r.threshold:.4f})")
        print("-" * 60)
        avg_f1 = df_res.f1_score.mean()
        print(f"AVERAGE F1: {avg_f1:.4f} trên {len(df_res)} placeId")

        # Lưu kết quả chi tiết và summary
        out_csv = 'results_weighted_ensemble.csv'
        df_res.to_csv(out_csv, index=False)
        summary_path = 'results_weighted_ensemble_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('TÓM TẮT WEIGHTED ENSEMBLE\n')
            f.write(f'Số placeId: {len(df_res)}\n')
            f.write(f'F1 trung bình: {avg_f1:.4f}\n')
            f.write(f'F1 cao nhất: {df_res.f1_score.max():.4f}\n')
            f.write(f'F1 thấp nhất: {df_res.f1_score.min():.4f}\n')
            f.write('\nTOP 5 theo F1:\n')
            top5 = df_res.sort_values('f1_score', ascending=False).head(5)
            for _, row in top5.iterrows():
                f.write(f"PlaceID {row.placeId}: F1={row.f1_score:.4f} True={row.num_true} Pred={row.num_pred} Len={row.len}\n")
        print(f"Đã lưu kết quả chi tiết: {out_csv} và summary: {summary_path}")

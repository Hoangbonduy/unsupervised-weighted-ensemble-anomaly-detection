import os
import warnings
# Tắt TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Chỉ hiện ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt oneDNN warnings
warnings.filterwarnings('ignore')

import anomaly_detection_base_model
import anomaly_injection
import borda_count_rank_aggregation
import model_centrality
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import get_prediction_error_ranking

# --- VÍ DỤ SỬ DỤNG ---
if __name__ == '__main__':
    # --- 1. THIẾT LẬP ĐƯỜNG DẪN VÀ THAM SỐ ---
    DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'labels/'
    
    # --- 2. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
    print("Đang đọc dữ liệu...")
    data_full = pd.read_csv(DATA_FILE_PATH)
    
    unique_place_ids = data_full['placeId'].unique()
    place_ids_to_process = unique_place_ids[:30]
    data_subset = data_full[data_full['placeId'].isin(place_ids_to_process)]
    
    # --- 3. LẶP QUA TỪNG PLACEID ĐỂ DỰ ĐOÁN VÀ ĐÁNH GIÁ ---
    print("\nBắt đầu quá trình dự đoán và đánh giá bằng Ensemble...")
    
    evaluation_results = []
    
    for place_id in tqdm(place_ids_to_process, desc="Đang xử lý PlaceIDs"):
        # Lấy dữ liệu chuỗi thời gian cho placeId hiện tại
        ts_group = data_subset[data_subset['placeId'] == place_id].sort_values('date')
        time_series_data = ts_group[['view']] # Giữ ở dạng DataFrame

        # Đọc file nhãn thật
        label_path = os.path.join(LABELS_DIR, f"label_{place_id}.csv")
        if not os.path.exists(label_path):
            print(f"Cảnh báo: Không tìm thấy file nhãn cho placeId {place_id}. Bỏ qua.")
            continue
            
        df_label = pd.read_csv(label_path)
        y_true = df_label['label'].to_numpy()

        # Kiểm tra sự khớp về độ dài
        if len(time_series_data) != len(y_true):
            print(f"Cảnh báo: Dữ liệu và nhãn cho placeId {place_id} không khớp độ dài. Bỏ qua.")
            continue

        # --- ĐOẠN CODE MỚI: ENSEMBLING BẰNG MEAN AGGREGATION ---

        # BƯỚC A: CHẠY TẤT CẢ CÁC MÔ HÌNH ĐỂ LẤY ĐIỂM SỐ
        # Tạo một dictionary để gọi các hàm lấy điểm số (scores)
        score_functions = {
            'SR': anomaly_detection_base_model.run_sr_scores,
            'IQR': anomaly_detection_base_model.run_iqr_scores,
            'MA': anomaly_detection_base_model.run_moving_average_scores,
            'IForest': anomaly_detection_base_model.run_iforest_scores,
            'MA_inverted': anomaly_detection_base_model.run_ma_inverted_scores,
            'KNN': anomaly_detection_base_model.run_knn_scores
        }

        all_scores = {}
        for name, score_func in score_functions.items():
            # Lấy điểm số từ mỗi mô hình
            scores = score_func(time_series_data)
            
            # Chuẩn hóa điểm số về thang 0-1 để có thể so sánh và cộng lại
            if np.std(scores) > 1e-9: # Kiểm tra để tránh chia cho 0
                scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                scores_normalized = scores # Nếu tất cả các điểm bằng nhau thì không cần chuẩn hóa
            
            all_scores[name] = scores_normalized

        # BƯỚC B: TỔNG HỢP ĐIỂM SỐ BẰNG CÁCH LẤY TRUNG BÌNH
        scores_df = pd.DataFrame(all_scores)
        aggregated_scores = scores_df.mean(axis=1).to_numpy() 

         # Ngưỡng theo contamination rate
        mean_score = np.mean(aggregated_scores)
        std_score = np.std(aggregated_scores)

        # Đặt ngưỡng là trung bình cộng với 2.5 lần độ lệch chuẩn
        # Bạn có thể thử nghiệm với số 2, 2.5 hoặc 3 để xem kết quả
        threshold = mean_score + 2.5 * std_score

        y_pred = (aggregated_scores > threshold).astype(int)

        selected_model_name = 'Ensemble_Mean'
        
    # --- KẾT THÚC ĐOẠN CODE ENSEMBLING ---

    # BƯỚC D: TÍNH TOÁN METRICS VÀ LƯU KẾT QUẢ
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    evaluation_results.append({
            'placeId': place_id,
            'selected_model': selected_model_name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'true_anomalies': np.sum(y_true),
            'pred_anomalies': np.sum(y_pred),
            'data_length': len(time_series_data)
        })

    # --- 4. HIỂN THỊ KẾT QUẢ TỔNG HỢP ---
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ PIPELINE ENSEMBLE MEAN")
    print("="*50)
    
    if not evaluation_results:
        print("Không có kết quả nào để hiển thị.")
    else:
        results_df = pd.DataFrame(evaluation_results)
        
        print("\n--- Chi tiết từng PlaceID ---")
        for _, row in results_df.iterrows():
            print(
                f"PlaceID {row['placeId']}: {row['selected_model']} -> F1={row['f1_score']:.3f} "
                f"P={row['precision']:.3f} R={row['recall']:.3f} "
                f"(True: {row['true_anomalies']}, Pred: {row['pred_anomalies']}, Len: {row['data_length']})"
            )
        
        print("\n--- Thống kê mô hình được chọn ---")
        print(results_df['selected_model'].value_counts())
        
        average_f1 = results_df['f1_score'].mean()
        average_precision = results_df['precision'].mean()
        average_recall = results_df['recall'].mean()

        print("\n" + "*"*50)
        print(f"==> AVERAGE METRICS TRÊN {len(results_df)} PLACEID: F1={average_f1:.4f} | Precision={average_precision:.4f} | Recall={average_recall:.4f}")
        print("*"*50)

        # Lưu kết quả
        try:
            out_csv = 'results_mean_ensemble.csv'
            results_df.to_csv(out_csv, index=False)
            with open('results_mean_ensemble_summary.txt', 'w', encoding='utf-8') as f:
                f.write('TÓM TẮT MEAN ENSEMBLE\n')
                f.write(f'Số placeId: {len(results_df)}\n')
                f.write(f'F1 trung bình: {average_f1:.4f}\n')
                f.write(f'Precision trung bình: {average_precision:.4f}\n')
                f.write(f'Recall trung bình: {average_recall:.4f}\n')
        except Exception as e:
            print(f'Không thể lưu file kết quả mean ensemble: {e}')
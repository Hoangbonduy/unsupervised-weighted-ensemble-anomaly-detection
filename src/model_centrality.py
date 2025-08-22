from scipy.stats import kendalltau
import numpy as np
import pandas as pd

def get_centrality_ranking(time_series, models_to_run):
    """Xếp hạng các mô hình dựa trên tính trung tâm (đã sửa lỗi nan)."""
    model_scores = {}
    X = pd.DataFrame({'value': time_series})
    
    for name, run_func in models_to_run.items():
        model_scores[name] = run_func(X)
        
    model_names = list(model_scores.keys())
    num_models = len(model_names)
    distance_matrix = np.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(i + 1, num_models):
            scores_i = model_scores[model_names[i]]
            scores_j = model_scores[model_names[j]]
            
            # <<< THAY ĐỔI CỐT LÕI: Kiểm tra trước khi tính kendalltau >>>
            # Nếu một trong hai mảng điểm số có phương sai gần bằng 0 (tức là hằng số)
            # thì không thể tính tương quan. Coi như chúng không tương quan (corr=0, dist=1).
            if np.std(scores_i) < 1e-9 or np.std(scores_j) < 1e-9:
                corr = 0.0
            else:
                try:
                    corr, _ = kendalltau(scores_i, scores_j)
                    # Kendall's Tau có thể trả về nan nếu có vấn đề khác
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
            
            distance = 1 - corr
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    avg_distances = np.mean(distance_matrix, axis=1)
    centrality_scores = {model_names[i]: avg_distances[i] for i in range(num_models)}
    
    # Khoảng cách càng thấp, mô hình càng trung tâm -> xếp hạng từ thấp đến cao
    ranked_models = sorted(centrality_scores, key=centrality_scores.get, reverse=False)
    
    return ranked_models, centrality_scores
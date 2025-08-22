import numpy as np

def borda_count_aggregation(rankings_list):
    if not rankings_list: return []
    
    all_models = list(rankings_list[0])
    num_models = len(all_models)
    borda_scores = {model: 0 for model in all_models}
    
    # Tạo một dict để lưu tất cả các thứ hạng của mỗi model
    model_ranks = {model: [] for model in all_models}
    
    for ranking in rankings_list:
        for i, model in enumerate(ranking):
            rank = i + 1
            borda_scores[model] += (num_models - rank)
            model_ranks[model].append(rank)
            
    # TÍNH TOÁN PHƯƠNG SAI THỨ HẠNG
    rank_variances = {model: np.var(ranks) for model, ranks in model_ranks.items()}
    
    # SẮP XẾP VỚI LUẬT PHỤ
    # Sắp xếp chính theo điểm Borda (cao hơn là tốt hơn)
    # Sắp xếp phụ theo phương sai thứ hạng (THẤP hơn là tốt hơn)
    final_ranking = sorted(
        all_models, 
        key=lambda model: (borda_scores[model], -rank_variances[model]), 
        reverse=True
    )
    
    # Trả về cả bảng xếp hạng và điểm số để debug
    return final_ranking, borda_scores
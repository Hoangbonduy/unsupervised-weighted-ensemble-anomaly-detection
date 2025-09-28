# analyze_features.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance():
    """
    Hàm này tải mô hình LightGBM đã huấn luyện và phân tích mức độ quan trọng
    của các meta-features.
    """
    print("="*80)
    print("Bắt đầu phân tích mức độ quan trọng của đặc trưng")
    print("="*80)

    # 1. Định nghĩa các file cần thiết
    model_file = 'lgbm_model_output/lgbm_classifier_model.joblib'
    metadata_file = 'lgbm_model_output/metadata.joblib'
    
    # Kiểm tra xem file model có tồn tại không
    try:
        model = joblib.load(model_file)
        metadata = joblib.load(metadata_file)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file model '{model_file}' hoặc metadata.")
        print("Vui lòng chạy lại script huấn luyện LightGBM trước.")
        return

    # 2. Lấy danh sách tên các đặc trưng và mức độ quan trọng của chúng
    feature_names = metadata['meta_feature_names']
    feature_importances = model.feature_importances_

    # Tạo một DataFrame để dễ dàng xem và sắp xếp
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    # 3. Sắp xếp các đặc trưng theo mức độ quan trọng giảm dần
    df_importance = df_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)

    # In ra các đặc trưng có mức độ quan trọng > 0
    important_features = df_importance[df_importance['importance'] > 0]
    
    if important_features.empty:
        print("!!! CẢNH BÁO: Không có đặc trưng nào có mức độ quan trọng > 0.")
        print("Điều này khẳng định rằng mô hình không học được gì từ dữ liệu.")
        return

    print(f"\nTìm thấy {len(important_features)}/{len(df_importance)} đặc trưng có mức độ quan trọng > 0.")
    
    print("\n--- Top 30 đặc trưng quan trọng nhất: ---")
    print(important_features.head(30))

    # 4. Vẽ biểu đồ để trực quan hóa
    print("\nĐang tạo biểu đồ top 20 đặc trưng quan trọng nhất...")
    
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='importance', 
        y='feature', 
        data=important_features.head(20)
    )
    plt.title('Top 20 Meta-Features quan trọng nhất')
    plt.xlabel('Mức độ quan trọng (Importance)')
    plt.ylabel('Tên Meta-Feature')
    plt.tight_layout()
    
    # Lưu biểu đồ ra file
    chart_path = 'lgbm_model_output/feature_importance.png'
    plt.savefig(chart_path)
    print(f"Đã lưu biểu đồ vào: '{chart_path}'")
    plt.show()


if __name__ == '__main__':
    analyze_feature_importance()
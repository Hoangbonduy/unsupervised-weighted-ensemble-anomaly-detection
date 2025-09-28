# extract_full_features.py

import pandas as pd
import numpy as np
from pycatch22 import catch22_all
from tsfresh import extract_features, settings
import warnings

warnings.filterwarnings('ignore')

def extract_full_meta_features(series_values):
    """
    Trích xuất tất cả 805 meta-features giống như trong quá trình training.
    Bao gồm 22 features từ catch22 và 783 features từ tsfresh.
    """
    
    # 1. Trích xuất 22 features từ catch22
    catch22_features = catch22_all(series_values)
    catch22_dict = {}
    for i, name in enumerate(catch22_features['names']):
        catch22_dict[name] = catch22_features['values'][i]
    
    # 2. Chuẩn bị dữ liệu cho tsfresh
    ts_df = pd.DataFrame({
        'id': [1] * len(series_values),
        'time': range(len(series_values)),
        'value': series_values
    })
    
    # 3. Trích xuất features từ tsfresh với settings đầy đủ
    extraction_settings = settings.EfficientFCParameters()
    tsfresh_features = extract_features(
        ts_df, 
        column_id='id', 
        column_sort='time',
        default_fc_parameters=extraction_settings,
        disable_progressbar=True
    )
    
    # 4. Kết hợp tất cả features
    all_features = {}
    
    # Thêm catch22 features
    for name, value in catch22_dict.items():
        all_features[name] = value
    
    # Thêm tsfresh features
    for col in tsfresh_features.columns:
        all_features[col] = tsfresh_features[col].iloc[0]
    
    # 5. Xử lý NaN và Inf
    for key, value in all_features.items():
        if pd.isna(value) or np.isinf(value):
            all_features[key] = 0.0
    
    return all_features

def test_feature_extraction():
    """Test function để kiểm tra việc trích xuất features"""
    # Tạo dữ liệu test
    test_series = np.random.randn(100)
    
    # Trích xuất features
    features = extract_full_meta_features(test_series)
    
    print(f"Số lượng features trích xuất được: {len(features)}")
    print("10 features đầu tiên:")
    for i, (name, value) in enumerate(features.items()):
        if i < 10:
            print(f"  {name}: {value}")
    
    return features

if __name__ == "__main__":
    test_feature_extraction()

#!/usr/bin/env python3
"""
Script kiểm tra số lượng mô hình trong BASE_MODELS
"""

import sys
import os

# Thêm thư mục src vào sys.path để import các module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    # Import các hàm chạy mô hình cơ sở từ thư mục src
    import anomaly_detection_base_model as ad_models
except ImportError as e:
    print(f"LỖI: Không thể import module: {e}")
    exit()

# Định nghĩa BASE_MODELS giống như trong 1_generate_performance_matrix.py
BASE_MODELS = {
    'SR': ad_models.run_sr_scores, 
    'MA': ad_models.run_moving_average_scores,
    'IForest': ad_models.run_iforest_scores, 
    'SH-ESD': ad_models.run_shesd_scores,
    'RePAD': ad_models.run_repad_scores,
    'Prophet': ad_models.run_prophet_scores, 
    'Moment': ad_models.run_moment_scores
}

print("🔍 KIỂM TRA SỐ LƯỢNG MÔ HÌNH")
print("="*50)
print(f"Tổng số mô hình trong BASE_MODELS: {len(BASE_MODELS)}")
print("\nDanh sách các mô hình:")
for i, (model_name, model_func) in enumerate(BASE_MODELS.items(), 1):
    print(f"  {i}. {model_name}")

print(f"\n⚠️ KẾT LUẬN:")
print(f"   - File 1_generate_performance_matrix.py: {len(BASE_MODELS)} mô hình")
print(f"   - File inception_regressor_rl.py mặc định: output_size=6")
print(f"   - KHÔNG KHỚP! Cần sửa output_size thành {len(BASE_MODELS)}")

# Kiểm tra xem các hàm có tồn tại không
print(f"\n🔧 KIỂM TRA CÁC HÀM:")
missing_functions = []
for model_name, model_func in BASE_MODELS.items():
    if hasattr(ad_models, model_func.__name__):
        print(f"  ✅ {model_name}: {model_func.__name__}")
    else:
        print(f"  ❌ {model_name}: {model_func.__name__} - KHÔNG TỒN TẠI")
        missing_functions.append(model_name)

if missing_functions:
    print(f"\n⚠️ CÁC HÀM THIẾU: {missing_functions}")
else:
    print(f"\n✅ TẤT CẢ CÁC HÀM ĐỀU TỒN TẠI")

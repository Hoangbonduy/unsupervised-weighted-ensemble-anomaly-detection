# inception_regressor_rl.py

import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, GlobalAveragePooling1D, Add, Concatenate, MaxPooling1D, Activation
from tensorflow.keras.models import Model
import time
import matplotlib.pyplot as plt
import numpy as np

class InceptionTimeRegressor:
    """
    Lớp InceptionTimeRegressor được sử dụng cho Representation Learning.
    Đầu vào: Chuỗi thời gian thô.
    Đầu ra: Dự đoán 7 giá trị VUS-ROC (SR, MA, IForest, SH-ESD, RePAD, Prophet, Moment).
    """
    def __init__(self, output_directory, input_shape, output_size=6, verbose=False,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, batch_size=16, nb_epochs=500, patience=50, 
                 initial_lr=0.001):

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.patience = patience
        self.verbose = verbose
        self.initial_lr = initial_lr

        self.model = self.build_model(input_shape, output_size)
        if verbose:
            self.model.summary()

    def _inception_module(self, input_tensor, stride=1, activation='relu'):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                    strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

        max_pool = MaxPooling1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        conv_max = Conv1D(filters=self.nb_filters, kernel_size=1,
                          padding='same', activation=activation, use_bias=False)(max_pool)
        conv_list.append(conv_max)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)
        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    def build_model(self, input_shape, output_size):
        input_layer = Input(input_shape)
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                input_res = self._shortcut_layer(input_res, x)
                x = input_res

        gap_layer = GlobalAveragePooling1D()(x)
        
        # Lớp đầu ra cho bài toán HỒI QUY (dự đoán 6 giá trị)
        output_layer = Dense(output_size, activation='linear')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_lr)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def fit(self, x_train, y_train, x_val, y_val):
        print(f"Bắt đầu huấn luyện với batch_size={self.batch_size}, epochs={self.nb_epochs}...")
        print(f"Learning rate ban đầu: {self.initial_lr}")
        
        # --- CALLBACKS ---
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.output_directory, 'best_model.h5'),
            monitor='val_loss', save_best_only=True, verbose=1)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1)
        
        # LEARNING RATE SCHEDULER - Giảm LR khi val_loss không cải thiện
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,           # Giảm LR xuống 50%
            patience=10,          # Chờ 10 epochs không cải thiện
            min_lr=1e-7,          # LR tối thiểu
            verbose=1,            # In thông báo khi LR thay đổi
            cooldown=0            # Số epochs chờ sau khi giảm LR
        )
        
        # COSINE ANNEALING SCHEDULER (tùy chọn - có thể bỏ comment để dùng)
        # cosine_scheduler = tf.keras.callbacks.CosineRestartScheduler(
        #     first_restart_epoch=50,
        #     t_mult=2.0,
        #     m_mult=0.9,
        #     alpha=0.0
        # )

        start_time = time.time()

        print("\n=== GIẢI THÍCH VAL LOSS PATTERN ===")
        print("🔍 Tại sao val_loss có thể TĂNG ở những epoch đầu rồi mới GIẢM?")
        print("1️⃣ KHỞI TẠO TRỌNG SỐ NGẪU NHIÊN:")
        print("   - Mô hình bắt đầu với trọng số ngẫu nhiên")
        print("   - Các dự đoán ban đầu có thể rất xa so với ground truth")
        print("2️⃣ LEARNING RATE CAO:")
        print("   - LR cao giúp thoát khỏi local minima nhưng có thể 'overshooting'")
        print("   - Gradient descent có thể 'nhảy qua' optimum trong những bước đầu")
        print("3️⃣ BATCH NORMALIZATION ADAPTION:")
        print("   - BatchNorm layers cần thời gian để học running mean/variance")
        print("   - Validation sử dụng running statistics chưa ổn định")
        print("4️⃣ OVERFITTING PHASE:")
        print("   - Train loss giảm nhanh, nhưng val loss tăng do overfitting")
        print("   - Regularization effects (dropout, weight decay) chưa phát huy")
        print("5️⃣ FEATURE LEARNING PHASE:")
        print("   - Mô hình đang học features từ đơn giản đến phức tạp")
        print("   - Val performance có thể dao động trước khi ổn định")
        print("="*60)

        hist = self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
            callbacks=[model_checkpoint, early_stopping, reduce_lr]
        )

        duration = time.time() - start_time
        print(f"\n✅ Huấn luyện hoàn tất trong {duration:.2f} giây.")
        
        # In thông tin về learning rate cuối cùng
        final_lr = self.model.optimizer.learning_rate.numpy()
        print(f"📊 Learning rate cuối cùng: {final_lr:.2e}")
        print(f"📈 Số epochs đã train: {len(hist.history['loss'])}")

        self.model.save(os.path.join(self.output_directory, 'last_model.h5'))
        self.model = tf.keras.models.load_model(os.path.join(self.output_directory, 'best_model.h5'))
        
        # Vẽ và lưu biểu đồ training history
        self.plot_training_history(hist)
        
        return hist

    def plot_training_history(self, history):
        """
        Vẽ và lưu biểu đồ train loss và validation loss
        """
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Lấy dữ liệu từ history
        epochs = range(1, len(history.history['loss']) + 1)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        # Vẽ biểu đồ
        ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        # Tìm epoch có val_loss thấp nhất
        min_val_loss_epoch = np.argmin(val_loss) + 1
        min_val_loss = min(val_loss)
        
        # Đánh dấu điểm val_loss thấp nhất
        ax.scatter([min_val_loss_epoch], [min_val_loss], 
                  color='red', s=100, zorder=5, marker='*')
        ax.annotate(f'Best Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch}',
                   xy=(min_val_loss_epoch, min_val_loss),
                   xytext=(min_val_loss_epoch + len(epochs)*0.1, min_val_loss + (max(val_loss) - min(val_loss))*0.1),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Cấu hình biểu đồ
        ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=14, fontweight='bold')
        ax.set_title('InceptionTime Training History\nTrain Loss vs Validation Loss', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Cải thiện appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=11)
        
        # Thêm thông tin tổng quan
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        total_epochs = len(epochs)
        
        info_text = f"""Training Summary:
Total Epochs: {total_epochs}
Final Train Loss: {final_train_loss:.4f}
Final Val Loss: {final_val_loss:.4f}
Best Val Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        plot_path = os.path.join(self.output_directory, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Lưu với định dạng PDF để in chất lượng cao
        plot_path_pdf = os.path.join(self.output_directory, 'training_history.pdf')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        
        print(f"\n📊 Biểu đồ Training History đã được lưu:")
        print(f"   - PNG: {plot_path}")
        print(f"   - PDF: {plot_path_pdf}")
        
        # Lưu training history dưới dạng CSV
        self.save_training_history_csv(history)
        
        plt.show()
        plt.close()

    def save_training_history_csv(self, history):
        """
        Lưu training history dưới dạng CSV để phân tích thêm
        """
        import pandas as pd
        
        # Tạo DataFrame từ history
        history_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })
        
        # Thêm các thông tin phân tích
        history_df['loss_diff'] = history_df['val_loss'] - history_df['train_loss']
        history_df['train_loss_change'] = history_df['train_loss'].pct_change().fillna(0)
        history_df['val_loss_change'] = history_df['val_loss'].pct_change().fillna(0)
        
        # Lưu file CSV
        csv_path = os.path.join(self.output_directory, 'training_history.csv')
        history_df.to_csv(csv_path, index=False)
        
        print(f"   - CSV: {csv_path}")
        
        # In thống kê tổng quan
        min_train_loss = history_df['train_loss'].min()
        min_val_loss = history_df['val_loss'].min()
        min_val_epoch = history_df.loc[history_df['val_loss'].idxmin(), 'epoch']
        
        print(f"\n📈 Thống kê Training:")
        print(f"   - Train Loss thấp nhất: {min_train_loss:.6f}")
        print(f"   - Validation Loss thấp nhất: {min_val_loss:.6f} (Epoch {min_val_epoch})")
        print(f"   - Tổng số epochs: {len(history_df)} epochs")
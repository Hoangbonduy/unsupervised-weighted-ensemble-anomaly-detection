# inception_regressor.py

import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, GlobalAveragePooling1D, Add, Concatenate, MaxPooling1D, Activation
from tensorflow.keras.models import Model
import time

class InceptionTimeRegressor:
    """
    Lớp InceptionTimeRegressor được điều chỉnh từ mã nguồn gốc của tác giả
    để thực hiện nhiệm vụ hồi quy đa đầu ra.
    """
    def __init__(self, output_directory, input_shape, output_size, verbose=False,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, batch_size=64, nb_epochs=200):

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.verbose = verbose

        self.model = self.build_model(input_shape, output_size)
        if verbose:
            self.model.summary()

    def _inception_module(self, input_tensor, stride=1, activation='relu'):
        """
        Hàm tạo một Inception Module, giữ nguyên logic từ mã nguồn gốc.
        """
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # Các kernel có kích thước giảm dần
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
        """
        Hàm tạo kết nối phần dư (shortcut/residual connection).
        """
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)
        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    def build_model(self, input_shape, output_size):
        """
        Xây dựng kiến trúc InceptionTime hoàn chỉnh cho bài toán hồi quy.
        """
        input_layer = Input(input_shape)
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                input_res = self._shortcut_layer(input_res, x)
                x = input_res

        gap_layer = GlobalAveragePooling1D()(x)
        
        # *** THAY ĐỔI QUAN TRỌNG: Lớp đầu ra cho hồi quy ***
        output_layer = Dense(output_size, activation='linear')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        
        # *** THAY ĐỔI QUAN TRỌNG: Compile với hàm mất mát MSE và learning rate thấp hơn ***
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)  # Thêm gradient clipping
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def fit(self, x_train, y_train, x_val, y_val):
        """
        Hàm huấn luyện mô hình.
        """
        print(f"Bắt đầu huấn luyện với batch_size={self.batch_size}, epochs={self.nb_epochs}...")
        
        # Callback để lưu lại model tốt nhất dựa trên val_loss
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.output_directory, 'best_model.h5'),
            monitor='val_loss', save_best_only=True)
        
        # Callback để dừng sớm (dừng sau 10 epochs không cải thiện)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        start_time = time.time()

        hist = self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
            callbacks=[model_checkpoint, early_stopping]
        )

        duration = time.time() - start_time
        print(f"Huấn luyện hoàn tất trong {duration:.2f} giây.")

        # Lưu model cuối cùng
        self.model.save(os.path.join(self.output_directory, 'last_model.h5'))

        # Tải lại model tốt nhất để đảm bảo chúng ta đang dùng nó
        self.model = tf.keras.models.load_model(os.path.join(self.output_directory, 'best_model.h5'))
        
        return hist
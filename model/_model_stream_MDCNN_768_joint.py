from tensorflow import keras as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.activations import relu, softmax


def BN_LeakyReLU(input):
    norm = BatchNormalization(axis=-1)(input)
    output = LeakyReLU(alpha=0.2)(norm)

    return output


def MDCN(input_layers, n_filters):
    '''
    Skip connection is added.
    '''
    # stream_left
    conv_left = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=4, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.0001))(input_layers)
    conv_left = BN_LeakyReLU(conv_left)
    # stream_middle_up
    conv_middle_1 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=3, kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(0.0001))(input_layers)
    conv_middle_1 = BN_LeakyReLU(conv_middle_1)
    # stream_right_up
    conv_right_1 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(input_layers)
    conv_right_1 = BN_LeakyReLU(conv_right_1)
    conv_right_2 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(conv_right_1)
    conv_right_2 = BN_LeakyReLU(conv_right_2)

    # stream_sum_1
    sum_1 = add([conv_middle_1, conv_right_2])

    # stream_middle_down
    conv_middle_2 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=3, kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(0.0001))(sum_1)
    conv_middle_2 = BN_LeakyReLU(conv_middle_2)
    # stream_right_down
    conv_right_3 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(sum_1)
    conv_right_3 = BN_LeakyReLU(conv_right_3)
    conv_right_4 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(conv_right_3)
    conv_right_4 = BN_LeakyReLU(conv_right_4)
    # print(input_layers.shape,conv_left.shape,conv_middle_2.shape,conv_right_4.shape)
    # stream_sum_2
    # sum_2 = add([conv_left, conv_middle_2, conv_right_4])
    sum_2 = add([conv_left, conv_middle_2, conv_right_4, input_layers])

    return sum_2


#
def DNCNN_vgg_768(input):  # 这里要注意  在主函数中已经声明过了
    n_filters = 48

    # input_layer = Input(shape=[224, 224, 3])
    # input_layer = Input(input_size)

    # ----- stage ----- 48
    conv_0 = Conv2D(n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(input)  # 48
    conv_0 = BN_LeakyReLU(conv_0)
    # print('conv_0: {}'.format(conv_0.shape))

    Max_pool_0 = MaxPooling2D(pool_size=2, strides=2, padding="valid")(conv_0)  # 遵循vgg19的设计
    # print('Max_pool_1: {}'.format(Max_pool_0.shape))

    # ----- stage ----- 48
    conv_1 = Conv2D(n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(Max_pool_0)  # 48
    conv_1 = BN_LeakyReLU(conv_1)
    # print('conv_1: {}'.format(conv_1.shape))
    conv_2 = Conv2D(n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(conv_1)  # 48
    conv_2 = BN_LeakyReLU(conv_2)
    # print('conv_2: {}'.format(conv_2.shape))

    Max_pool_1 = MaxPooling2D(pool_size=2, strides=2, padding="valid")(conv_2)  # 遵循vgg19的设计
    # print('Max_pool_1: {}'.format(Max_pool_1.shape))

    #  ----- stage ----- 96
    conv_3 = Conv2D(2 * n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(Max_pool_1)  # 96
    conv_3 = BN_LeakyReLU(conv_3)
    # print('conv_2: {}'.format(conv_3.shape))
    conv_4 = Conv2D(2 * n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(conv_3)  # 96
    conv_4 = BN_LeakyReLU(conv_4)
    # print('conv_3: {}'.format(conv_4.shape))

    Max_pool_2 = MaxPooling2D(pool_size=2, strides=2, padding="valid")(conv_4)  # 遵循vgg19的设计 ===== 1
    # print('Max_pool_2: {}'.format(Max_pool_2.shape))

    # ----- stage ----- 192
    conv_5 = Conv2D(4 * n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(Max_pool_2)  # 192
    conv_5 = BN_LeakyReLU(conv_5)
    # print('conv_4: {}'.format(conv_5.shape))
    block_1 = MDCN(conv_5, 4 * n_filters)  # 192                                                 =====
    # print('block_1: {}'.format(block_1.shape))

    Max_pool_3 = MaxPooling2D(pool_size=2, strides=2, padding="valid")(block_1)  # 遵循vgg19的设计 ===== 2
    # print('Max_pool_3: {}'.format(Max_pool_3.shape))

    # ----- stage ----- 384
    conv_6 = Conv2D(8 * n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(Max_pool_3)  # 384
    conv_6 = BN_LeakyReLU(conv_6)
    # print('conv_6: {}'.format(conv_6.shape))
    block_2 = MDCN(conv_6, 8 * n_filters)  # 384                                                 =====
    # print('block_2: {}'.format(block_2.shape))

    Max_pool_4 = MaxPooling2D(pool_size=2, strides=2, padding="valid")(block_2)  # 遵循vgg19的设计 ===== 3
    # print('Max_pool_4: {}'.format(Max_pool_4.shape))

    # ----- stage ----- 768
    conv_7 = Conv2D(16 * n_filters, (3, 3), padding='same', strides=(1, 1), kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(Max_pool_4)  # 768
    conv_7 = BN_LeakyReLU(conv_7)
    # print('conv_7: {}'.format(conv_7.shape))

    block_3 = MDCN(conv_7, 16 * n_filters)  # 768                                                =====
    # print('block_3: {}'.format(block_3.shape))

    # # ----------
    # gap = GlobalAveragePooling2D()(block_3)  # GAP
    # fc = Dense(1536, activation='relu')(gap)  # FC
    # fc = Dropout(0.5)(fc)  # Dropout
    # fc = Dense(1536, activation='relu')(fc)  # FC
    # fc = Dropout(0.5)(fc)  # Dropout
    # fc = Dense(768, activation='relu')(fc)  # FC
    # output = Dropout(0.5)(fc)  # Dropout
    #
    # predictions = Dense(2, activation='sigmoid')(output)  # sigmoid
    #
    # model_DNCNN = Model(inputs=input_layer, outputs=predictions)
    #
    # optm = K.optimizers.Adam(learning_rate=1e-4)
    #
    # model_DNCNN.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['acc'])

    return conv_4, block_1, block_2, block_3
    # return block_2

# input_size = (224, 224, 3)
# import tensorflow as tf
#
# input_shape = (1, 224, 224, 3)
# x = tf.random.normal(input_shape)
# model = DNCNN_vgg_768(input_size=[224,224,3])  #
# # print("model_CNN_stream.layers[-3].output:", model.layers[-3].output)
# # print("model summary:", model.summary())
# print(model.summary())

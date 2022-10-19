import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import tensorflow.keras.layers as L
import tensorflow.keras as K
from net_utils._model_stream_Swin_Transformer__joint import Swin_MDCNN_Joint_base
from net_utils._model_stream_MDCNN_768_joint import DNCNN_vgg_768


def Fusion_JAG(X_Swin_Stage, L_swin, C_swin, X_MDCNN_Stage, H_mdcnn, W_mdcnn, C_mdcnn):
    # Fusion Stage 3
    X_Swin_Stage3_flatten = L.Flatten()(X_Swin_Stage)
    X_MDCNN_Stage3_flatten = L.Flatten()(X_MDCNN_Stage)

    X_StackFeature = tf.stack([X_Swin_Stage3_flatten, X_MDCNN_Stage3_flatten], 2)
    X_StackFeature = tf.expand_dims(X_StackFeature, axis=1)

    conv_1 = Conv2D(256, (1, 1), padding='same')(X_StackFeature)

    leftWeight = Conv2D(1, (1, 1), padding='same')(conv_1)
    rightWeight = Conv2D(1, (1, 1), padding='same')(conv_1)
    leftWeight = tf.squeeze(leftWeight, axis=[1, 3])
    rightWeight = tf.squeeze(rightWeight, axis=[1, 3])

    concat_Swin = L.Multiply()([X_Swin_Stage3_flatten, leftWeight])
    concat_MDCNN = L.Multiply()([X_MDCNN_Stage3_flatten, rightWeight])

    concat_Swin_ = tf.reshape(concat_Swin, shape=[-1, L_swin, C_swin])
    concat_MDCNN_ = tf.reshape(concat_MDCNN, shape=[-1, H_mdcnn, W_mdcnn, C_mdcnn])

    return concat_Swin, concat_MDCNN, concat_Swin_, concat_MDCNN_


# 4 Stage Fusion Method
def Swin_MDCNN_Joint_4_JAG_Skip(input_size, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96,
                                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                                qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=LayerNormalization, patch_norm=True, num_class=2):
    # num_layers = len(depths)
    # patches_resolution = [img_size[0] // patch_size[0],
    #                       img_size[1] // patch_size[1]]
    # dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

    # define Input_layer
    IN = Input(input_size)

    # Swin-Transfomer Stream stage 4 output
    # # (None, 7*7, 768) 没有经过 patch merging 操作的Swin_Transformer的输出
    # _X0, X0_, _X1, X1_, _X2, X2_, _X3
    _, _, _, _, _, _, _X3 = Swin_MDCNN_Joint_base(IN, img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                                  embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                  window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                  qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                  drop_path_rate=drop_path_rate, norm_layer=norm_layer, patch_norm=patch_norm)
    X_Swin_Stage4 = _X3
    _, L_swin, C_swin = X_Swin_Stage4.get_shape().as_list()

    # MDCNN Stream stage 4 output
    # # (None, 7, 7, 768)
    # conv_4, block_1, block_2, block_3
    _, _, _, block_3 = DNCNN_vgg_768(IN)
    X_MDCNN_Stage4 = block_3
    _, H_mdcnn, W_mdcnn, C_mdcnn = X_MDCNN_Stage4.get_shape().as_list()

    # Fusion operation
    # (None, 196, 384) / (None, 14, 14, 384)
    concat_Swin, concat_MDCNN, _, _ = Fusion_JAG(X_Swin_Stage4, L_swin, C_swin, X_MDCNN_Stage4, H_mdcnn, W_mdcnn, C_mdcnn)
    #
    concat_Swin_MDCNN = tf.stack([concat_Swin, concat_MDCNN], 2)
    concat_Swin_MDCNN = tf.expand_dims(concat_Swin_MDCNN, axis=1)

    # BN for CAM
    # concat_Swin_MDCNN = BatchNormalization(axis=-1, name='CAM_BN_1')(concat_Swin_MDCNN)  # compute BN channel dim
    # print('------conv_Swin_MDCNN_BN {}'.format(concat_Swin_MDCNN.shape))
    # concat_Swin_MDCNN = K.activations.relu(concat_Swin_MDCNN)
    # print('------conv_Swin_MDCNN_relu {}'.format(concat_Swin_MDCNN.shape))
    # # BN for CAM

    conv_Swin_MDCNN = Conv2D(384, (1, 1), padding='same', kernel_initializer="he_normal",
                             kernel_regularizer=regularizers.l2(1e-4), name='conv_CAM')(concat_Swin_MDCNN)

    conv_Swin_MDCNN = BatchNormalization(axis=-1, name='CAM_BN_2')(conv_Swin_MDCNN)
    conv_Swin_MDCNN = K.activations.relu(conv_Swin_MDCNN)

    gap = GlobalAveragePooling2D()(conv_Swin_MDCNN)
    print("gap_shape", gap.shape)

    # get prediction
    fc = L.Dense(192)(gap)
    print("gap_shape", fc.shape)
    fc = L.Dropout(0.5)(fc)
    predictions = L.Dense(num_class, activation="sigmoid")(fc)

    model = K.models.Model(inputs=IN, outputs=predictions)
    optm = K.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optm, loss=['categorical_crossentropy'], metrics=['acc'])

    return model

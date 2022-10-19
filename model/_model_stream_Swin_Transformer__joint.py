# Source: https://github.com/rishigami/Swin-Transformer-TF

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D, Input
from tensorflow.keras import regularizers
import tensorflow.keras.layers as L
import tensorflow.keras as K

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2],
                           num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2],
                           num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2],
                           num_heads=[6, 12, 24, 48])
}


class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                             window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (
                                                                            2 * self.window_size[1] - 1),
                                                                self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                          None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
            (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization,
                 prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    prefix=self.prefix)
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        # print("x_get type H {}--W {} ------------------".format(H, W))
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # print("-----------------SwinTransformerBlock x-------------------",x.get_shape())

        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x


# class BasicLayer(tf.keras.layers.Layer):  # SwinTransformerBlock + PatchMerging
#     """
#     补
#     A basic Swin Transformer layer for one stage.
#     Args:
#         dim (int): Number of input channels.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         downsample (layer.Layer | None, optional): Downsample layer at the end of the layer. Default: None
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#         self.list_ = []
#
#         # build blocks
#         self.blocks = tf.keras.Sequential([
#             SwinTransformerBlock(dim=dim,
#                                  input_resolution=input_resolution,
#                                  num_heads=num_heads,
#                                  window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias,
#                                  qk_scale=qk_scale,
#                                  drop=drop,
#                                  attn_drop=attn_drop,
#                                  drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob,
#                                                                                 list) else drop_path_prob,
#                                  norm_layer=norm_layer,
#                                  prefix=f'{prefix}/blocks{i}')
#             for i in range(depth)])
#
#         if downsample is not None:
#             self.downsample = downsample(
#                 input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
#         else:
#             self.downsample = None
#
#     def call(self, x):
#         x = self.blocks(x)
#
#         # print("---------------------------x_shape-----------------------", x.get_shape())
#
#         if self.downsample is not None:
#             x_down = self.downsample(x)
#         else:
#             x_down = None
#
#         # print("---------------------------x_downsample_shape-----------------------", x.get_shape())
#
#         return x, x_down  # 改


class PatchEmbed(tf.keras.layers.Layer):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__(name='patch_embed')  ##
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]  # 56*56个 每个patch大小是4*4
        self.img_size = img_size  # 224*224
        self.patch_size = patch_size  # patch_size = 4*4
        self.patches_resolution = patches_resolution  # resolution[224,224]  -->  resolution[4,4]
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # num_patch = 56*56

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj')
        if norm_layer is not None:  # is not None --> LayerNormalization
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])  # embed_dim
        if self.norm is not None:
            x = self.norm(x)
        return x


# ================================================以下为自行构造==================================================
# 先手动构造BasicLayer
def BasicLayer(X, dim, input_resolution, depth, num_heads, window_size,
               mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
               drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, prefix=''):
    # biuld blocks
    for i in range(depth):  # depth 需要定义 这只是一个数字
        # print("transformer depth ?  ?------------------| {}".format(depth))
        # print("get transformer block ------------------| {}".format(i))
        X = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                         i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path_prob=drop_path_prob[i] if isinstance(
                                     drop_path_prob, list) else drop_path_prob,
                                 norm_layer=norm_layer,
                                 prefix=f'{prefix}/blocks{i}')(X)

    if downsample is not None:
        X_downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)(X)  # patch merging
    else:
        X_downsample = None

    return X, X_downsample


def Swin_MDCNN_Joint_base(X, img_size=(224, 224), patch_size=(4, 4), in_chans=3,
                          embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                          window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                          norm_layer=LayerNormalization, patch_norm=True):
    num_layers = len(depths)
    # stochastic depth
    dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

    X = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                   norm_layer=norm_layer if patch_norm else None)(X)
    # num_patches = X.num_patches
    patches_resolution = [img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1]]
    # patches_resolution = X.patches_resolution
    X = Dropout(drop_rate)(X)

    Layer_output = []

    # _X Originel feature from transformer; X_ Patch Merging feature (down_sample feature)
    # stage 1
    i_layer = 0
    _X0, X0_ = BasicLayer(X, dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          window_size=window_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                              depths[:i_layer + 1])],
                          norm_layer=norm_layer,
                          downsample=PatchMerging if (
                                  i_layer < num_layers - 1) else None,
                          prefix=f'layers{i_layer}')

    # print("---------------Transformer block get feature shape {}".format(_X0.get_shape()))
    # print("---------------Patch merging get feature shape {}".format(X0_.get_shape()))

    # stage 2
    i_layer = 1
    _X1, X1_ = BasicLayer(X0_, dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          window_size=window_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                              depths[:i_layer + 1])],
                          norm_layer=norm_layer,
                          downsample=PatchMerging if (
                                  i_layer < num_layers - 1) else None,
                          prefix=f'layers{i_layer}')

    # print("---------------Transformer block get feature shape {}".format(_X1.get_shape()))
    # print("---------------Patch merging get feature shape {}".format(X1_.get_shape()))

    # stage 3
    i_layer = 2
    _X2, X2_ = BasicLayer(X1_, dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          window_size=window_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                              depths[:i_layer + 1])],
                          norm_layer=norm_layer,
                          downsample=PatchMerging if (
                                  i_layer < num_layers - 1) else None,
                          prefix=f'layers{i_layer}')

    # print("---------------Transformer block get feature shape {}".format(_X2.get_shape()))
    # print("---------------Patch merging get feature shape {}".format(X2_.get_shape()))

    # stage 4
    i_layer = 3
    _X3, X3_ = BasicLayer(X2_, dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          window_size=window_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                              depths[:i_layer + 1])],
                          norm_layer=norm_layer,
                          downsample=PatchMerging if (
                                  i_layer < num_layers - 1) else None,
                          prefix=f'layers{i_layer}')
    #
    # print("---------------Transformer block get feature shape {}".format(_X3.get_shape()))
    # print("---------------Patch merging get feature shape {}".format(X3_))
    #
    # X = _X3
    #
    # X = norm_layer(epsilon=1e-5, name='norm')(X)
    # X = GlobalAveragePooling1D()(X)

    return _X0, X0_, _X1, X1_, _X2, X2_, _X3
    # return _X2


# def Swin_MDCNN_Joint(input_size):
#     NUM_CLASS = 2
#
#     IN = Input(input_size)
#
#     # base
#     X = Swin_MDCNN_Joint_base(IN, img_size=(224, 224), patch_size=(4, 4), in_chans=3,
#                               embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                               window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                               norm_layer=LayerNormalization, patch_norm=True)
#
#     X = L.Dense(512, activation='relu')(X)  # FC
#     X = L.Dense(256, activation='relu')(X)  # FC
#     predictions = L.Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_normal',
#                           kernel_regularizer=regularizers.l2(0.0001))(X)  # softmax
#
#     model = K.models.Model(inputs=IN, outputs=predictions)
#
#     optm = K.optimizers.Adam(learning_rate=1e-4)
#     model.compile(optimizer=optm, loss=['categorical_crossentropy'], metrics=['acc'])
#
#     return model
#
#
# model = Swin_MDCNN_Joint(input_size=[224, 224, 3])
# print(model.summary())


# # class SwinTransformerModel(tf.keras.Model):
# class SwinTransformerModel(tf.keras.layers.Layer):
#     r"""
#     补
#     Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#     Args:
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """
#
#     def __init__(self, model_name='swin_tiny_patch4_window7_224', include_top=False,
#                  img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=LayerNormalization, ape=False, patch_norm=True,
#                  use_checkpoint=False, **kwargs):
#         super().__init__(name=model_name)  ##
#
#         # self.include_top = include_top  # 分类头
#
#         # self.num_classes = num_classes  # 类别个数
#         self.num_layers = len(depths)  # 控制stage数量  swin中有4个stage，融合的关键就是将这四个stage取出
#         self.embed_dim = embed_dim  # 根据设置赋值 默认使用tiny:96
#         self.ape = ape  # absolute positional embedding 绝对位置编码 // 默认是不使用的 文中用的应该是相对位置编码
#         self.patch_norm = patch_norm  # If True, add normalization after patch embedding. Default: True
#         # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # ? 似乎没有用到?
#         self.mlp_ratio = mlp_ratio  # Ratio of mlp hidden dim to embedding dim. Default: 4
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(img_size=img_size,
#                                       patch_size=patch_size,
#                                       in_chans=in_chans,
#                                       embed_dim=embed_dim,
#                                       norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
#                                                       shape=(1, num_patches, embed_dim),
#                                                       initializer=tf.initializers.Zeros())
#
#         self.pos_drop = Dropout(drop_rate)
#
#         # stochastic depth
#         dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]
#
#         # build layers
#         # self.basic_layers = tf.keras.Sequential([
#         #     BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#         #                input_resolution=(patches_resolution[0] // (2 ** i_layer),
#         #                                  patches_resolution[1] // (2 ** i_layer)),
#         #                depth=depths[i_layer],
#         #                num_heads=num_heads[i_layer],  # 设置好的
#         #                window_size=window_size,
#         #                mlp_ratio=self.mlp_ratio,
#         #                qkv_bias=qkv_bias,
#         #                qk_scale=qk_scale,
#         #                drop=drop_rate,
#         #                attn_drop=attn_drop_rate,
#         #                drop_path_prob=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#         #                norm_layer=norm_layer,
#         #                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#         #                use_checkpoint=use_checkpoint,
#         #                prefix=f'layers{i_layer}')
#         #     for i_layer in range(self.num_layers)])
#         #     #| 这里用一个循环，用来生成4个basic_layer |
#
#         self.i_0_layer = 0
#         self.basic_layers_0 = BasicLayer(dim=int(embed_dim * 2 ** self.i_0_layer),
#                                          input_resolution=(patches_resolution[0] // (2 ** self.i_0_layer),
#                                                            patches_resolution[1] // (2 ** self.i_0_layer)),
#                                          depth=depths[self.i_0_layer],  # 2
#                                          num_heads=num_heads[self.i_0_layer],  # 3
#                                          window_size=window_size,
#                                          mlp_ratio=self.mlp_ratio,
#                                          qkv_bias=qkv_bias,
#                                          qk_scale=qk_scale,
#                                          drop=drop_rate,
#                                          attn_drop=attn_drop_rate,
#                                          drop_path_prob=dpr[
#                                                         sum(depths[:self.i_0_layer]):sum(depths[:self.i_0_layer + 1])],
#                                          norm_layer=norm_layer,
#                                          downsample=PatchMerging if (self.i_0_layer < self.num_layers - 1) else None,
#                                          # 由于这里进行了PatchMerging操作，所以直接就没有了[56,56,96]的输出
#                                          use_checkpoint=use_checkpoint,
#                                          prefix=f'layers{self.i_0_layer}')
#         self.i_1_layer = 1
#         self.basic_layers_1 = BasicLayer(dim=int(embed_dim * 2 ** self.i_1_layer),
#                                          input_resolution=(patches_resolution[0] // (2 ** self.i_1_layer),
#                                                            patches_resolution[1] // (2 ** self.i_1_layer)),
#                                          depth=depths[self.i_1_layer],
#                                          num_heads=num_heads[self.i_1_layer],  # 设置好的
#                                          window_size=window_size,
#                                          mlp_ratio=self.mlp_ratio,
#                                          qkv_bias=qkv_bias,
#                                          qk_scale=qk_scale,
#                                          drop=drop_rate,
#                                          attn_drop=attn_drop_rate,
#                                          drop_path_prob=dpr[
#                                                         sum(depths[:self.i_1_layer]):sum(depths[:self.i_1_layer + 1])],
#                                          norm_layer=norm_layer,
#                                          downsample=PatchMerging if (self.i_1_layer < self.num_layers - 1) else None,
#                                          use_checkpoint=use_checkpoint,
#                                          prefix=f'layers{self.i_1_layer}')
#         self.i_2_layer = 2
#         self.basic_layers_2 = BasicLayer(dim=int(embed_dim * 2 ** self.i_2_layer),
#                                          input_resolution=(patches_resolution[0] // (2 ** self.i_2_layer),
#                                                            patches_resolution[1] // (2 ** self.i_2_layer)),
#                                          depth=depths[self.i_2_layer],
#                                          num_heads=num_heads[self.i_2_layer],  # 设置好的
#                                          window_size=window_size,
#                                          mlp_ratio=self.mlp_ratio,
#                                          qkv_bias=qkv_bias,
#                                          qk_scale=qk_scale,
#                                          drop=drop_rate,
#                                          attn_drop=attn_drop_rate,
#                                          drop_path_prob=dpr[
#                                                         sum(depths[:self.i_2_layer]):sum(depths[:self.i_2_layer + 1])],
#                                          norm_layer=norm_layer,
#                                          downsample=PatchMerging if (self.i_2_layer < self.num_layers - 1) else None,
#                                          use_checkpoint=use_checkpoint,
#                                          prefix=f'layers{self.i_2_layer}')
#         self.i_3_layer = 3
#         self.basic_layers_3 = BasicLayer(dim=int(embed_dim * 2 ** self.i_3_layer),
#                                          input_resolution=(patches_resolution[0] // (2 ** self.i_3_layer),
#                                                            patches_resolution[1] // (2 ** self.i_3_layer)),
#                                          depth=depths[self.i_3_layer],
#                                          num_heads=num_heads[self.i_3_layer],  # 设置好的
#                                          window_size=window_size,
#                                          mlp_ratio=self.mlp_ratio,
#                                          qkv_bias=qkv_bias,
#                                          qk_scale=qk_scale,
#                                          drop=drop_rate,
#                                          attn_drop=attn_drop_rate,
#                                          drop_path_prob=dpr[
#                                                         sum(depths[:self.i_3_layer]):sum(depths[:self.i_3_layer + 1])],
#                                          norm_layer=norm_layer,
#                                          downsample=PatchMerging if (self.i_3_layer < self.num_layers - 1) else None,
#                                          use_checkpoint=use_checkpoint,
#                                          prefix=f'layers{self.i_3_layer}')
#
#         # 如果需要basiclayer的输出数据，那就要在basiclayer里面来定义一个列表，将每一层的输出都保存在相应的列表里，然后传出来？
#
#         self.norm = norm_layer(epsilon=1e-5, name='norm')
#         self.avgpool = GlobalAveragePooling1D()
#         # if self.include_top:
#         #     self.head = Dense(num_classes, name='head')
#         # else:
#         #     self.head = None
#
#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         # x = self.basic_layers(x)  # 4-stages
#         # _x: without downsample, x_0, x_1, x_2
#         _0x, x_0 = self.basic_layers_0(x)  # Original: (None, 3136, 96)  56*56 , Down: (None, 784, 192)
#         print("basic_layers_0, Original: {}, Down: {}".format(_0x.get_shape(), x_0.get_shape()))
#         _1x, x_1 = self.basic_layers_1(x_0)  # Original: (None, 784, 192)  28*28 , Down: (None, 196, 384)
#         print("basic_layers_1, Original: {}, Down: {}".format(_1x.get_shape(), x_1.get_shape()))
#         _2x, x_2 = self.basic_layers_2(x_1)  # Original: (None, 196, 384)  14*14  , Down: (None, 49, 768)
#         print("basic_layers_2, Original: {}, Down: {}".format(_2x.get_shape(), x_2.get_shape()))
#         _3x, x_3 = self.basic_layers_3(x_2)  # Original: (None, 49, 768)   7*7, Down: None
#         print("basic_layers_3, Original: {}, Down: {}".format(_3x.get_shape(), x_3))
#
#         # print("-----------------basic_layers-----------------",x.get_shape())
#         x = _3x
#         # x = self.norm(x)   # 是否要经过avgpool操作之后再连接？
#         # x = self.avgpool(x)  # end
#         return _2x, _3x
#
#     def call(self, x):
#         _2x, _3x = self.forward_features(x)
#         # if self.include_top:
#         #     x = self.head(x)
#         return _2x, _3x


# # def transmit():
#
#
# def SwinTransformer(model_name='swin_tiny_224', num_classes=1000, include_top=True, pretrained=True, use_tpu=False,
#                     cfgs=CFGS):
#     cfg = cfgs[model_name]
#     net = SwinTransformerModel(
#         model_name=model_name, include_top=include_top, num_classes=num_classes, img_size=cfg['input_size'],
#         window_size=cfg[
#             'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
#     )
#     net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))  # 输入
#     if pretrained is True:
#         url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
#         pretrained_ckpt = tf.keras.utils.get_file(
#             model_name, url, untar=True)
#     else:
#         pretrained_ckpt = pretrained
#
#     if pretrained_ckpt:
#         if tf.io.gfile.isdir(pretrained_ckpt):
#             pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'
#
#         if use_tpu:
#             load_locally = tf.saved_model.LoadOptions(
#                 experimental_io_device='/job:localhost')
#             net.load_weights(pretrained_ckpt, options=load_locally)
#         else:
#             net.load_weights(pretrained_ckpt)
#
#     return net
#
#
# def get_Swin_Transformer():
#     NUM_CLASS = 2
#     base_model = SwinTransformer(model_name='swin_tiny_224', num_classes=NUM_CLASS, include_top=False, pretrained=False)
#     x = base_model.output
#     x = L.Dense(128, activation='relu')(x)  # FC
#     predictions = L.Dense(NUM_CLASS, activation='softmax', kernel_initializer='he_normal',
#                           kernel_regularizer=regularizers.l2(0.0001))(x)  # softmax
#     model_Swin = K.models.Model(inputs=base_model.input, outputs=predictions)
#
#     optm = K.optimizers.Adam(learning_rate=1e-4)
#     model_Swin.compile(optimizer=optm, loss=['categorical_crossentropy'], metrics=['acc'])
#
#     return model_Swin
#
#
# import tensorflow as tf
#
# # input_shape = (1, 224, 224, 3)
# # x = tf.random.normal(input_shape)
# model_ViT = get_Swin_Transformer()
# print(model_ViT.output)
# print("model summary:", model_ViT.summary())

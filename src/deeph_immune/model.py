"""DeepH-Immune V2 shortcut-resistant model.

Canonical GitHub implementation used to rebuild the architecture before loading
the supplied fold checkpoints. The layer names and tensor shapes are intentionally
kept identical to the training implementation.

Serializable shortcut-resistant DeepH-Immune .

This version is computationally equivalent to the shortcut-resistant model,
but replaces anonymous Keras Lambda layers with explicit serializable layers.

Why:
Keras 3 may fail to reload .keras files containing Lambda layers whose
output shape cannot be inferred during deserialization.

Existing trained .keras models DO NOT need to be retrained:
for external validation, rebuild this architecture and call model.load_weights()
on each existing .keras checkpoint.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, Input, layers, regularizers

PAD_ID = 0
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}
VOCAB_SIZE = len(AA_VOCAB) + 1
PEP_MAX_LEN = 25
MHC_ALIGNED_LEN = 281
ESM_DIM = 1280


def tokenize_peptide(seq: str, max_len: int = PEP_MAX_LEN):
    s = "".join(str(seq).split()).upper()
    if not (1 <= len(s) <= max_len):
        raise ValueError(f"Peptide length {len(s)} outside 1..{max_len}: {seq!r}")
    ids = [AA_TO_ID.get(aa, AA_TO_ID["X"]) for aa in s]
    return ids + [PAD_ID] * (max_len - len(ids))


def tokenize_mhc_aligned(seq: str, expected_len: int = MHC_ALIGNED_LEN):
    s = "".join(str(seq).split()).upper().replace("-", "*").replace(".", "*")
    if len(s) != expected_len:
        raise ValueError(
            f"MHC aligned sequence must have length {expected_len}, got {len(s)}"
        )
    return [
        PAD_ID if aa == "*" else AA_TO_ID.get(aa, AA_TO_ID["X"])
        for aa in s
    ]


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class TokenMask(layers.Layer):
    def call(self, tokens):
        return tf.cast(tf.not_equal(tokens, PAD_ID), tf.float32)


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class LearnedPositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.pos_emb = layers.Embedding(self.max_len, self.d_model)

    def call(self, x):
        positions = tf.range(0, tf.shape(x)[1])
        return x + self.pos_emb(positions)[None, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, x):
        return tf.math.l2_normalize(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class AbsDifference(layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class MaskedAttentionPooling(layers.Layer):
    def __init__(self, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.proj = layers.Dense(self.hidden_dim, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def call(self, inputs):
        x, mask = inputs
        logits = tf.squeeze(self.score(self.proj(x)), axis=-1)
        logits = tf.where(
            tf.cast(mask, tf.bool),
            logits,
            tf.cast(-1e9, logits.dtype),
        )
        weights = tf.nn.softmax(logits, axis=1)
        weights = weights * tf.cast(mask, weights.dtype)
        weights = weights / (
            tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-8
        )
        return tf.reduce_sum(x * weights[:, :, None], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_dim": self.hidden_dim})
        return cfg


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class CoreWindowAttention(layers.Layer):
    def __init__(self, core_len=9, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.core_len = int(core_len)
        self.hidden_dim = int(hidden_dim)
        self.proj = layers.Dense(self.hidden_dim, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def call(self, inputs):
        x, mask = inputs
        frames = tf.signal.frame(
            x, frame_length=self.core_len, frame_step=1, axis=1
        )
        mask_frames = tf.signal.frame(
            mask, frame_length=self.core_len, frame_step=1, axis=1
        )
        valid = tf.reduce_all(mask_frames > 0.5, axis=-1)
        window_repr = tf.reduce_mean(frames, axis=2)

        logits = tf.squeeze(self.score(self.proj(window_repr)), axis=-1)
        logits = tf.where(valid, logits, tf.cast(-1e9, logits.dtype))
        weights = tf.nn.softmax(logits, axis=1)
        weights = weights * tf.cast(valid, weights.dtype)
        weights = weights / (
            tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-8
        )
        return tf.reduce_sum(window_repr * weights[:, :, None], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "core_len": self.core_len,
            "hidden_dim": self.hidden_dim,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class CCTMaskedEncoder(layers.Layer):
    def __init__(
        self,
        max_len,
        d_model=64,
        conv_kernel=3,
        num_heads=4,
        num_blocks=1,
        ffn_mult=2,
        dropout=0.20,
        l2_reg=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.conv_kernel = int(conv_kernel)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.ffn_mult = int(ffn_mult)
        self.dropout_rate = float(dropout)
        self.l2_reg = float(l2_reg)

        reg = regularizers.l2(self.l2_reg)
        self.embedding = layers.Embedding(
            VOCAB_SIZE, self.d_model, embeddings_regularizer=reg
        )
        self.pos = LearnedPositionalEmbedding(self.max_len, self.d_model)
        self.conv = layers.Conv1D(
            self.d_model,
            kernel_size=self.conv_kernel,
            padding="same",
            activation=tf.nn.gelu,
            kernel_regularizer=reg,
        )
        self.input_dropout = layers.Dropout(self.dropout_rate)

        self.norm1, self.mha, self.drop1 = [], [], []
        self.norm2, self.ffn1, self.ffn2, self.drop2 = [], [], [], []

        for i in range(self.num_blocks):
            self.norm1.append(
                layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_{i}")
            )
            self.mha.append(
                layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=max(1, self.d_model // self.num_heads),
                    dropout=self.dropout_rate,
                    name=f"self_attn_{i}",
                )
            )
            self.drop1.append(layers.Dropout(self.dropout_rate))
            self.norm2.append(
                layers.LayerNormalization(epsilon=1e-6, name=f"ln_ffn_{i}")
            )
            self.ffn1.append(
                layers.Dense(
                    self.d_model * self.ffn_mult,
                    activation=tf.nn.gelu,
                    kernel_regularizer=reg,
                )
            )
            self.ffn2.append(
                layers.Dense(self.d_model, kernel_regularizer=reg)
            )
            self.drop2.append(layers.Dropout(self.dropout_rate))

    def call(self, tokens, training=None):
        mask = tf.cast(tf.not_equal(tokens, PAD_ID), tf.float32)
        mask3 = mask[:, :, None]

        x = self.embedding(tokens)
        x = self.pos(x)
        x = x * tf.cast(mask3, x.dtype)
        x = self.conv(x)
        x = x * tf.cast(mask3, x.dtype)
        x = self.input_dropout(x, training=training)

        mask_bool = tf.cast(mask, tf.bool)
        attn_mask = tf.logical_and(
            mask_bool[:, :, None],
            mask_bool[:, None, :],
        )

        for i in range(self.num_blocks):
            h = self.norm1[i](x)
            attn = self.mha[i](
                h, h,
                attention_mask=attn_mask,
                training=training,
            )
            x = x + self.drop1[i](attn, training=training)
            x = x * tf.cast(mask3, x.dtype)

            h = self.norm2[i](x)
            h = self.ffn1[i](h)
            h = self.ffn2[i](h)
            x = x + self.drop2[i](h, training=training)
            x = x * tf.cast(mask3, x.dtype)

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
            "conv_kernel": self.conv_kernel,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "ffn_mult": self.ffn_mult,
            "dropout": self.dropout_rate,
            "l2_reg": self.l2_reg,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class CrossAttentionWithScores(layers.Layer):
    def __init__(self, d_model=64, num_heads=4, dropout=0.20, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)

        self.q_norm = layers.LayerNormalization(epsilon=1e-6)
        self.kv_norm = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.d_model // self.num_heads),
            dropout=self.dropout_rate,
        )
        self.drop = layers.Dropout(self.dropout_rate)
        self.out_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        query, key_value, query_mask, kv_mask = inputs

        q = self.q_norm(query)
        kv = self.kv_norm(key_value)

        q_bool = tf.cast(query_mask, tf.bool)
        kv_bool = tf.cast(kv_mask, tf.bool)
        cross_mask = tf.logical_and(
            q_bool[:, :, None],
            kv_bool[:, None, :],
        )

        attended_mhc, scores = self.mha(
            query=q,
            value=kv,
            key=kv,
            attention_mask=cross_mask,
            return_attention_scores=True,
            training=training,
        )

        out = query + self.drop(attended_mhc, training=training)
        out = self.out_norm(out)
        out = out * tf.cast(query_mask[:, :, None], out.dtype)
        return out, scores

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="DeepHImmuneShortcutResistant")
class GatedFusion(layers.Layer):
    def __init__(self, fusion_dim=256, dropout=0.25, l2_reg=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.fusion_dim = int(fusion_dim)
        self.dropout_rate = float(dropout)
        self.l2_reg = float(l2_reg)

        reg = regularizers.l2(self.l2_reg)
        self.seq_proj = layers.Dense(
            self.fusion_dim,
            activation=tf.nn.gelu,
            kernel_regularizer=reg,
        )
        self.esm_proj = layers.Dense(
            self.fusion_dim,
            activation=tf.nn.gelu,
            kernel_regularizer=reg,
        )
        self.gate = layers.Dense(
            self.fusion_dim,
            activation="sigmoid",
            kernel_regularizer=reg,
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        seq_feat, esm_feat = inputs
        s = self.seq_proj(seq_feat)
        e = self.esm_proj(esm_feat)
        g = self.gate(tf.concat([s, e], axis=-1))
        fused = g * s + (1.0 - g) * e
        return self.drop(self.norm(fused), training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "fusion_dim": self.fusion_dim,
            "dropout": self.dropout_rate,
            "l2_reg": self.l2_reg,
        })
        return cfg


def build_deeph_immune_v2_shortcut_resistant(
    pep_len=PEP_MAX_LEN,
    mhc_len=MHC_ALIGNED_LEN,
    esm_dim=ESM_DIM,
    d_model=64,
    num_heads=4,
    dropout=0.25,
    l2_reg=1e-5,
    learning_rate=3e-4,
):
    pep_input = Input((pep_len,), dtype="int32", name="pep_tokens")
    mhc_input = Input((mhc_len,), dtype="int32", name="mhc_tokens")
    esm_pep_input = Input((esm_dim,), dtype="float32", name="esm_peptide")
    esm_mhc_input = Input((esm_dim,), dtype="float32", name="esm_mhc")

    pep_mask = TokenMask(name="pep_mask")(pep_input)
    mhc_mask = TokenMask(name="mhc_mask")(mhc_input)

    pep_encoded = CCTMaskedEncoder(
        max_len=pep_len,
        d_model=d_model,
        conv_kernel=3,
        num_heads=num_heads,
        num_blocks=2,
        dropout=dropout,
        l2_reg=l2_reg,
        name="peptide_cct_encoder",
    )(pep_input)

    mhc_encoded = CCTMaskedEncoder(
        max_len=mhc_len,
        d_model=d_model,
        conv_kernel=5,
        num_heads=num_heads,
        num_blocks=1,
        dropout=dropout,
        l2_reg=l2_reg,
        name="mhc_cct_encoder",
    )(mhc_input)

    cross_out, cross_scores = CrossAttentionWithScores(
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="pep_to_mhc_cross_attention",
    )([pep_encoded, mhc_encoded, pep_mask, mhc_mask])

    cross_pool = MaskedAttentionPooling(
        hidden_dim=d_model,
        name="cross_attention_pool",
    )([cross_out, pep_mask])

    pep_pool = MaskedAttentionPooling(
        hidden_dim=d_model,
        name="peptide_pool",
    )([pep_encoded, pep_mask])

    core_pool = CoreWindowAttention(
        core_len=9,
        hidden_dim=d_model,
        name="candidate_9mer_core_pool",
    )([pep_encoded, pep_mask])

    seq_features = layers.Concatenate(
        name="sequence_feature_concat"
    )([cross_pool, core_pool, pep_pool])

    reg = regularizers.l2(l2_reg)

    esm_pep = layers.Dense(
        128,
        activation=tf.nn.gelu,
        kernel_regularizer=reg,
        name="esm_pep_bottleneck",
    )(esm_pep_input)

    esm_mhc = layers.Dense(
        128,
        activation=tf.nn.gelu,
        kernel_regularizer=reg,
        name="esm_mhc_bottleneck",
    )(esm_mhc_input)

    esm_pep_norm = L2Normalize(
        axis=-1,
        name="esm_pep_l2norm",
    )(esm_pep)

    esm_mhc_norm = L2Normalize(
        axis=-1,
        name="esm_mhc_l2norm",
    )(esm_mhc)

    esm_product = layers.Multiply(
        name="esm_pmhc_product"
    )([esm_pep_norm, esm_mhc_norm])

    esm_absdiff = AbsDifference(
        name="esm_pmhc_absdiff"
    )([esm_pep_norm, esm_mhc_norm])

    esm_features = layers.Concatenate(
        name="esm_feature_concat"
    )([
        esm_pep_norm,
        esm_product,
        esm_absdiff,
    ])

    esm_features = layers.Dense(
        128,
        activation=tf.nn.gelu,
        kernel_regularizer=reg,
        name="esm_interaction_bottleneck",
    )(esm_features)
    esm_features = layers.Dropout(dropout)(esm_features)

    fused = GatedFusion(
        fusion_dim=256,
        dropout=dropout,
        l2_reg=l2_reg,
        name="gated_sequence_esm_fusion",
    )([seq_features, esm_features])

    x = layers.Dense(
        128,
        activation=tf.nn.gelu,
        kernel_regularizer=reg,
        name="classifier_dense_128",
    )(fused)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        64,
        activation=tf.nn.gelu,
        kernel_regularizer=reg,
        name="classifier_dense_64",
    )(x)
    x = layers.Dropout(0.25)(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="immunogenicity",
    )(x)

    inputs = [
        pep_input,
        mhc_input,
        esm_pep_input,
        esm_mhc_input,
    ]

    model = Model(
        inputs=inputs,
        outputs=output,
        name="DeepH_Immune_V2_ShortcutResistant",
    )

    attention_model = Model(
        inputs=inputs,
        outputs=cross_scores,
        name="DeepH_Immune_V2_ShortcutResistant_CrossAttention",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auroc"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    return model, attention_model


CUSTOM_OBJECTS = {
    "TokenMask": TokenMask,
    "LearnedPositionalEmbedding": LearnedPositionalEmbedding,
    "L2Normalize": L2Normalize,
    "AbsDifference": AbsDifference,
    "MaskedAttentionPooling": MaskedAttentionPooling,
    "CoreWindowAttention": CoreWindowAttention,
    "CCTMaskedEncoder": CCTMaskedEncoder,
    "CrossAttentionWithScores": CrossAttentionWithScores,
    "GatedFusion": GatedFusion,
}

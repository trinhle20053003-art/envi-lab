"""BiLSTM Encoder + LSTM Decoder + additive Bahdanau Attention bằng Keras."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="EnVi")
class PaddingMask(layers.Layer):
    """True = token thật; False = token PAD (mã 0)."""
    def call(self, token_ids):
        return tf.not_equal(token_ids, 0)


@keras.utils.register_keras_serializable(package="EnVi")
class BahdanauAttention(layers.Layer):
    """Decoder tìm phần câu nguồn cần chú ý trước khi dự đoán token tiếp theo."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query_projection = layers.Dense(units, use_bias=False)
        self.key_projection = layers.Dense(units, use_bias=False)
        self.score_projection = layers.Dense(1, use_bias=False)
        self.supports_masking = True

    def build(self, input_shape):
        self.query_projection.build(input_shape[0])
        self.key_projection.build(input_shape[1])
        self.score_projection.build((None, None, None, self.units))
        super().build(input_shape)

    def call(self, inputs):
        decoder_outputs, encoder_outputs, source_mask = inputs
        # decoder_outputs: [batch, số token Việt, decoder_hidden]
        # encoder_outputs: [batch, số token Anh, 2 * encoder_hidden]
        query = self.query_projection(decoder_outputs)
        key = self.key_projection(encoder_outputs)
        scores = self.score_projection(
            tf.nn.tanh(query[:, :, None, :] + key[:, None, :, :])
        )
        scores = tf.squeeze(scores, axis=-1)

        # PAD phải nhận xác suất 0 sau softmax.
        scores = tf.where(source_mask[:, None, :], scores, -1e9)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attention_weights, encoder_outputs)
        return context, attention_weights

    def compute_mask(self, inputs, mask=None):
        return None, None

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.utils.register_keras_serializable(package="EnVi")
class RemoveMask(layers.Layer):
    """Loss tự bỏ PAD bằng nhãn; không để Keras áp dụng thêm mask lần thứ hai."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return tf.identity(inputs)

    def compute_mask(self, inputs, mask=None):
        return None


@keras.utils.register_keras_serializable(package="EnVi")
def masked_loss(target_ids, logits):
    token_losses = keras.losses.sparse_categorical_crossentropy(
        target_ids, logits, from_logits=True
    )
    real_tokens = tf.cast(tf.not_equal(target_ids, 0), token_losses.dtype)
    return tf.math.divide_no_nan(
        tf.reduce_sum(token_losses * real_tokens), tf.reduce_sum(real_tokens)
    )


@keras.utils.register_keras_serializable(package="EnVi")
class TokenNLL(keras.metrics.Metric):
    """Cross-entropy trung bình trên token thật của cả tập, không tính PAD."""
    def __init__(self, name="token_nll", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_sum = self.add_weight(name="loss_sum", initializer="zeros")
        self.token_count = self.add_weight(name="token_count", initializer="zeros")

    def update_state(self, target_ids, logits, sample_weight=None):
        losses = keras.losses.sparse_categorical_crossentropy(target_ids, logits, from_logits=True)
        valid = tf.cast(tf.not_equal(target_ids, 0), tf.float32)
        self.loss_sum.assign_add(tf.reduce_sum(tf.cast(losses, tf.float32) * valid))
        self.token_count.assign_add(tf.reduce_sum(valid))

    def result(self):
        return tf.math.divide_no_nan(self.loss_sum, self.token_count)

    def reset_state(self):
        self.loss_sum.assign(0.0)
        self.token_count.assign(0.0)

def build_model(source_vocab_size, target_vocab_size, config):
    """Dùng Functional API để xem model.summary() và lưu thành tệp .keras."""
    source_tokens = keras.Input(shape=(None,), dtype="int32", name="source_tokens")
    previous_target_tokens = keras.Input(shape=(None,), dtype="int32", name="previous_target_tokens")
    source_mask = PaddingMask(name="source_mask")(source_tokens)
    target_mask = PaddingMask(name="target_mask")(previous_target_tokens)

    source_embedding = layers.Embedding(
        source_vocab_size, config["embedding_dim"], name="source_embedding"
    )(source_tokens)
    source_embedding = layers.Dropout(config["dropout"])(source_embedding)

    # Hai LSTM đọc câu nguồn theo hai chiều. Ghép trạng thái để tạo decoder_hidden.
    encoder = layers.Bidirectional(
        layers.LSTM(config["encoder_units"], return_sequences=True, return_state=True),
        name="bilstm_encoder",
    )
    memory, forward_h, forward_c, backward_h, backward_c = encoder(
        source_embedding, mask=source_mask
    )
    initial_h = layers.Concatenate(name="initial_h")([forward_h, backward_h])
    initial_c = layers.Concatenate(name="initial_c")([forward_c, backward_c])

    target_embedding_layer = layers.Embedding(
        target_vocab_size, config["embedding_dim"], name="target_embedding"
    )
    target_embeddings = target_embedding_layer(previous_target_tokens)
    target_embeddings = layers.Dropout(config["dropout"])(target_embeddings)
    decoder = layers.LSTM(
        2 * config["encoder_units"], return_sequences=True, return_state=True,
        name="lstm_decoder",
    )
    decoder_outputs, _, _ = decoder(
        target_embeddings, initial_state=[initial_h, initial_c], mask=target_mask
    )
    context, _ = BahdanauAttention(config["attention_units"], name="bahdanau_attention")(
        [decoder_outputs, memory, source_mask]
    )
    combined = layers.Concatenate(name="context_and_decoder")([decoder_outputs, context])
    combined = layers.Dense(config["embedding_dim"], activation="tanh", name="attention_fusion")(combined)
    combined = layers.Dropout(config["dropout"], name="output_dropout")(combined)
    logits = layers.Dense(target_vocab_size, name="vocabulary_logits", dtype="float32")(combined)
    logits = RemoveMask(name="unmasked_logits")(logits)

    return keras.Model([source_tokens, previous_target_tokens], logits, name="ENVI_BiLSTM_Attention")


def build_inference_models(model):
    """Dùng lại đúng trọng số model đã train; không tạo model dịch mới."""
    encoder_model = keras.Model(
        model.inputs[0],
        [RemoveMask()(model.get_layer("bilstm_encoder").output[0]),
         model.get_layer("initial_h").output,
         model.get_layer("initial_c").output,
         model.get_layer("source_mask").output],
        name="inference_encoder",
    )
    hidden_size = model.get_layer("lstm_decoder").units
    token_input = keras.Input(shape=(1,), dtype="int32")
    memory_input = keras.Input(shape=(None, hidden_size), dtype="float32")
    h_input = keras.Input(shape=(hidden_size,), dtype="float32")
    c_input = keras.Input(shape=(hidden_size,), dtype="float32")
    mask_input = keras.Input(shape=(None,), dtype="bool")

    embeddings = model.get_layer("target_embedding")(token_input)
    outputs, next_h, next_c = model.get_layer("lstm_decoder")(
        embeddings, initial_state=[h_input, c_input]
    )
    context, weights = model.get_layer("bahdanau_attention")([outputs, memory_input, mask_input])
    combined = layers.Concatenate()([outputs, context])
    combined = model.get_layer("attention_fusion")(combined)
    logits = model.get_layer("vocabulary_logits")(combined)
    decoder_model = keras.Model(
        [token_input, memory_input, h_input, c_input, mask_input],
        [logits, next_h, next_c, weights], name="inference_decoder_step",
    )
    return encoder_model, decoder_model

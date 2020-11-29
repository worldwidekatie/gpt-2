import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.compat.v1.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.compat.v1.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.compat.v1.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.compat.v1.cond(
       tf.compat.v1.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.compat.v1.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.compat.v1.cumsum(tf.compat.v1.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.compat.v1.stack([
        tf.compat.v1.range(0, batch),
        # number of indices to include
        tf.compat.v1.maximum(tf.compat.v1.reduce_sum(tf.compat.v1.cast(cumulative_probs <= p, tf.compat.v1.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.compat.v1.gather_nd(sorted_logits, indices)
    return tf.compat.v1.where(
        logits < min_values,
        tf.compat.v1.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.compat.v1.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.compat.v1.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.compat.v1.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.compat.v1.multinomial(logits, num_samples=1, output_dtype=tf.compat.v1.int32)
            return [
                next_outputs['presents'] if past is None else tf.compat.v1.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.compat.v1.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.compat.v1.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.compat.v1.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.compat.v1.TensorShape([batch_size, None]),
                tf.compat.v1.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens

import numpy as np
import pytest
import tensorflow as tf

from lib.model import (
    build_attention_over_sequence_layer,
    generate_mask_matrix,
    replace_nan_by_zero,
)
from lib.preprocessing import transform_image_embeddings


@pytest.mark.parametrize(
    "x,expected",
    [
        (
            tf.constant([[np.nan, 1.0, 5.0], [3.0, 2.0, 4.0]]),
            tf.constant([[0, 1.0, 5.0], [3.0, 2.0, 4.0]]),
        ),
        (
            tf.constant([[np.nan, np.nan, np.nan], [3.0, 2.0, 4.0]]),
            tf.constant([[0, 0, 0], [3.0, 2.0, 4.0]]),
        ),
        (np.random.rand(10, 2, 20), None),
    ],
)
def test_replace_nan_by_zero(x, expected):
    if expected is None:
        expected = x
    assert (replace_nan_by_zero(x) == expected).numpy().all()


def test_generate_mask_matrix():
    input_tensor = np.array([[1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0]], dtype=int)
    output = generate_mask_matrix(input_tensor).numpy()
    assert output.shape == (2, 6, 6)
    assert (np.transpose(output, axes=(0, 2, 1)) == output).all()


def test_build_attention_over_sequence_layer():
    embedding_dim = 512
    input_name = "image_embedding"
    inputs, outputs = build_attention_over_sequence_layer(
        embedding_dim, input_name, num_heads=1, key_dim=64
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    batch_size = 3
    sequence_dim = 10
    embeddings = np.random.rand(batch_size, sequence_dim, embedding_dim)
    mask = np.zeros((batch_size, sequence_dim))
    mask[0, :] = 1
    mask[2, 0] = 1
    attention_output, attention_scores, average_output = model([embeddings, mask])
    assert attention_output.shape == (batch_size, sequence_dim, embedding_dim)
    # Only one attention head
    assert attention_scores.shape == (batch_size, 1, sequence_dim, sequence_dim)
    # As we masked 2nd element in batch, scores should be uniformly spread

    assert (
        attention_scores.numpy()[1]
        == np.ones((1, sequence_dim, sequence_dim), dtype=np.float32) / sequence_dim
    ).all()
    assert (
        attention_scores.numpy()[2, 0, 0]
        == np.array([1.0] + [0.0] * 9, dtype=np.float32)
    ).all()
    assert (
        attention_scores.numpy()[2, 0, 1:]
        == np.ones((sequence_dim - 1, sequence_dim), dtype=np.float32) / sequence_dim
    ).all()
    assert average_output.shape == (batch_size, embedding_dim)
    # as we masked all embeddings for 2nd element in batch, we expect an
    # average output of zero
    assert (average_output.numpy()[1] == 0.0).all()


def test_transform_image_embeddings():
    embedding_dim = 512
    max_images = 10
    image_embeddings = {i: np.ones(embedding_dim, dtype=np.float32) for i in range(6)}
    output = transform_image_embeddings(image_embeddings, max_images, embedding_dim)
    assert output.shape == (max_images, embedding_dim)
    assert (output[6:] == 0.0).all()
    assert (output[:6] == 1.0).all()

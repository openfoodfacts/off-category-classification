"""Heleprs for training a model
without penalizing it for predicting categories compatible with dataset
(like the one given by taxonomy.Taxonomy.cats_compat)
"""
import tensorflow as tf


class TaxonomyTransformer:
    """Provides transformers to be applied to dataset

    :param .taxonomy.Taxonomy taxo: a taxonomy instance
    """

    def __init__(self, taxo):
        self.taxo = taxo

    @property
    def add_ancestors(self):
        """return a add_ancestor transformer"""

        taxo = self.taxo

        def add_ancestors(ds: tf.data.Dataset):
            """Add ancestors to y to be sure to have coherent data

            Note: this does not seems the right way to go, 
            we prefer to use add_compatible_categories and MaskingModel
            """

            def get_ancestors(cats):
                cats_list = [c.decode("utf-8") for c in cats.numpy()]
                result = list(taxo.cats_complete(taxo.cats_filter(cats_list)))
                return tf.constant(result, dtype=tf.string)

            def _transform(x, y):
                y = tf.py_function(func=get_ancestors, inp=[y], Tout=tf.string)
                # make shape clear to avoid failing at compile time on next steps
                y = tf.reshape(y, [-1])
                return (x, y)

            # apply to dataset
            return (
                ds
                .map(_transform, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            )

        return add_ancestors

    @property
    def add_compatible_categories(self):
        """return a add_compatible_categories transformer"""

        taxo = self.taxo

        def add_compatible_categories(ds: tf.data.Dataset):
            """Add a specific feature with compatible categories"""
            def get_compat(cats):
                cats_list = [c.decode("utf-8") for c in cats.numpy()]
                result = list(taxo.cats_compat(taxo.cats_filter(cats_list)))
                return tf.constant(result, dtype=tf.string)

            def _transform(x, y):
                x["compat"] = tf.py_function(func=get_compat, inp=[y], Tout=tf.string)
                # make shape clear to avoid failing at compile time on next steps
                x["compat"] = tf.reshape(x["compat"], [-1])
                return (x, y)
            # apply to dataset
            return (
                ds
                .map(_transform, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            )

        return add_compatible_categories


def binarize_compat(x, categories_multihot, feat_name="compat"):
    """Binarize the compatibility mask

    we must do that as we binarize y (in categories_encode)
    so that mask is consistent with y
    """
    if feat_name not in x:
        return x  # nothing to do
    # vector where compat categories are 1
    binarized = categories_multihot(x["compat"])
    binarized = binarized[1:]  # drop OOV
    # make it a mask where compatibles are 0, the rest one
    binarized = tf.ones(binarized.shape, dtype=tf.float32) - binarized
    # make a new ds (we can't modify current x)
    return dict(
        x,
        compat=binarized,
    )


class MaskingModel(tf.keras.Model):
    """A model that applies a mask taken
    from features to y_pred before computing loss

    Note that to predict the model does not need this feature.

    :param str mask_feature: tells which feature to use for mask
    """
    def __init__(self, *args, mask_feature="compat", **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_feature = mask_feature

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # first mask categories we do not want using the compat feature
        mask = x[self.mask_feature]
        # zeros mask_feature in y_pred, as we know they are not in y_pred
        y_pred *= mask
        return super().compute_loss(x, y, y_pred, sample_weight)
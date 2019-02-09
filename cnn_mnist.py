import tensorflow as tf
import numpy as np
import optimizers

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):

    inputs = tf.reshape(
        tensor=features["x"],
        shape=[-1, 28, 28, 1]
    )

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2]
    )

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2]
    )

    inputs = tf.reshape(
        tensor=inputs,
        shape=[-1, 7 * 7 * 64]
    )

    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024,
        activation=tf.nn.relu
    )

    inputs = tf.layers.dropout(
        inputs=inputs,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = optimizers.SantaOptimizer(
            eta=4e-11,
            gamma=0.8,
            sigma=0.8,
            const=1000,
            epsilon=1e-5,
            burnin=10000
        )

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


def main(unused_argv):

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = np.asarray(mnist.train.images, dtype=np.float32)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = np.asarray(mnist.test.images, dtype=np.float32)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="mnist_convnet_model"
    )

    mnist_classifier.train(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True
        ),
        steps=20000,
        hooks=[
            tf.train.LoggingTensorHook(
                tensors={"probabilities": "softmax_tensor"},
                every_n_iter=50
            )
        ]
    )

    eval_results = mnist_classifier.evaluate(
        input_fn=tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )
    )

    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

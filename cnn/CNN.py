import tensorflow as tf
import numpy as np


def write_and_encode(data_list, tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for label, data_matrix in data_list.items():
        example = tf.train.Example(features=tf.train.Features (
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "data_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_matrix.tostring()]))
            }
        ))
        writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(tfrecord_filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([tfrecord_filename],)
    _, serialized_example = reader.read(filename_queue)
    feature = tf.parse_single_example(serialized_example,
                                      features={
                                          "label": tf.FixedLenFeature([], tf.int64),
                                          "data_raw": tf.FixedLenFeature([], tf.string)
                                      })
    data = tf.decode_raw(feature["data_raw"], tf.int32)
    data = tf.reshape(data, [2, 5])
    return feature["label"], data


def convolution_layer(input_layer, num_filters, kernel_size_matrix, activation_method):
    conv_layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=num_filters,
        kernel_size=kernel_size_matrix,
        padding="same",
        activation=activation_method
    )
    return conv_layer


def pooling_layer(input_layer, pool_size_matrix, num_strides):
    pooling_layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size_matrix,
        strides=num_strides
    )
    return pooling_layer


def full_connect_layer(input_layer, num_units, activation_method=None):
    dense = tf.layers.dense(
        inputs=input_layer,
        units=num_units,
        activation=activation_method
    )
    return dense


def construct_cnn(input_layer, mode):

    with tf.variable_scope("conv1"):
        num_filters = 15
        kernel_size = [5, 5]
        activation_method = tf.nn.relu

        pool_size_matrix = [2, 2]
        stride = 2

        conv1 = convolution_layer(input_layer, num_filters, kernel_size, activation_method)
        pool1 = pooling_layer(conv1, pool_size_matrix, stride)

    with tf.variable_scope("conv2"):
        num_filters = 20
        kernel_size = [5, 5]
        activation_method = tf.nn.relu

        pool_size_matrix = [2, 2]
        stride = 2

        conv2 = convolution_layer(pool1, num_filters, kernel_size, activation_method)
        pool2 = pooling_layer(conv2, pool_size_matrix, stride)

    with tf.variable_scope("dense"):
        pool2_flat = tf.reshape(pool2, [-1])

        num_units = 1000
        activation_method = tf.nn.relu
        dense = full_connect_layer(pool2_flat, num_units, activation_method)
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=(mode==tf.estimator.ModeKeys.TRAIN)
        )

    with tf.variable_scope("logits"):
        output_units = 5
        logits = full_connect_layer(dropout, output_units)

    return logits



def main():
    data_list = dict()
    data_list[1] = np.arange(1, 11).reshape(2, 5)
    data_list[2] = np.arange(10, 20).reshape(2, 5)
    # 读入数据
    tf_filename = "test.tfrecord"
    write_and_encode(data_list, tf_filename)
    labels, data = read_and_decode(tf_filename)

    # 构建CNN模型，参数在模块中修改
    mode = tf.estimator.ModeKeys.TRAIN
    logits = construct_cnn(data, mode=mode)

    predictions = {
        "classes": tf.arg_max(input=logits, axis=1)
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # 构建模型
    model_fn = None
    # 预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        model_fn = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        model_fn = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
    # 评估模型
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }
        model_fn = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )

    cnn_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="tmp/AR_minist_convnet_model"
    )

    # 记录log
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

# sess = tf.InteractiveSession()
    # print(label.eval(), data.eval())
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(2):
            label_1, data_1 = sess.run([label, data])
            print("label = ", label_1, "\ndata = \n", data_1)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
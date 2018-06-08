import tensorflow as tf
import numpy as np
from utils import readers
from utils import pre_precessing
from cnn.app_flag import FLAGS

def write_and_encode(data_list, tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for label, data_matrix in data_list:
        example = tf.train.Example(features=tf.train.Features(
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
    data = tf.decode_raw(feature["data_raw"], tf.float64)
    data = tf.reshape(data, [FLAGS.image_rows, FLAGS.image_cols])
    return data, feature["label"]


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


def train_input_fn():

    tfrecord_file = "../resources/train_tfrecord"  # 训练数据文件路径
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parser)

    train_dataset = dataset.repeat(FLAGS.num_epochs).batch(FLAGS.batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()

    features, labels = train_iterator.get_next()
    return features, labels


def parser(record_line):

    features = {
        "label": tf.FixedLenFeature([], tf.int64),
        "data_raw": tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record_line, features=features)
    label = tf.cast(parsed["label"], tf.int32)
    data = tf.decode_raw(parsed["data_raw"], tf.float64)
    data = tf.reshape(data, [FLAGS.image_rows, FLAGS.image_cols])
    data = tf.cast(data, tf.float32)
    return data, label


def eval_input_fn():

    tfrecord_file = "../resources/test_tfrecord"  # 测试数据文件路径
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parser)
    num_epochs = 5
    batch_size = 5

    eval_dataset = dataset.repeat(num_epochs).batch(batch_size)
    eval_iterator = eval_dataset.make_one_shot_iterator()

    features, labels = eval_iterator.get_next()
    return features, labels


def cnn_fn(features, labels, mode):

    with tf.variable_scope("input"):
        input_layer = tf.reshape(features, [-1, FLAGS.image_rows, FLAGS.image_cols, 1])  # 将输入转化为一定维度的向量和矩阵

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
        pool2_flat = tf.reshape(pool2, [-1, 9*165*20])
        print(pool2_flat)
        num_units = 1000
        activation_method = tf.nn.relu
        dense = full_connect_layer(pool2_flat, num_units, activation_method)
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=(mode==tf.estimator.ModeKeys.TRAIN)
        )

    with tf.variable_scope("logits"):
        output_units = FLAGS.label_size
        logits = full_connect_layer(dropout, output_units)

    predictions = {
        "classes": tf.arg_max(input=logits, dimension=1, name="classes"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # 构建模型方程
    model_fn = None
    # 预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        model_fn = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32)-1, depth=FLAGS.label_size)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, scope="LOSS")

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

    return model_fn


def write_user_instances_to_tfrecord():
    """
    将所有的数据文件组织成tfrecord格式并写入resources中，instance是float64格式，label是int64格式
    :return:
    """
    users = ["0"+str(i) for i in range(1, 10)]
    users.extend([str(i) for i in range(10, 17)])
    users.extend(["32", "40", "41", "42", "43", "49", "50", "51"])

    # 读取所有用户的所有数据，以（label, instance）的形式放入instances
    instances = []
    for user in users:
        train_data = readers.read_user_files(user)
        for label, instance in train_data.items():
            instances.append((label, instance))

    # 预处理所有数据，例如将所有数据扩展到同一个维度
    formalized_instances = pre_precessing.extend_to_maxsize(instances)

    # 将数据写入tfrecord用来做训练和测试
    train_instances = formalized_instances[:100]
    write_and_encode(train_instances, "../resources/train_tfrecord")

    test_instances = formalized_instances[101:]
    write_and_encode(test_instances, "../resources/test_tfrecord")


def main():

    # 构建模型
    cnn_classifier = tf.estimator.Estimator(
        model_fn=cnn_fn, model_dir="tmp/AR_minist_convnet_model"
    )
    cnn_classifier.train(input_fn=train_input_fn)
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)

    # 记录log
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50
    # )


def main2():
    instance, label = read_and_decode("../resources/test_tfrecord")
    instance_batch, label_batch = tf.train.shuffle_batch([instance, label], batch_size=3,
                                                         capacity=200, min_after_dequeue=100, num_threads=2)
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    i_, l_ = sess.run([instance_batch, label_batch])
    print("instance_batch:", i_, "\nlabel_batch:", l_)


if __name__ == "__main__":
    main()
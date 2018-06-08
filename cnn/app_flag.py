import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_path", "../resources", "data path")
tf.app.flags.DEFINE_integer("image_rows", 39, "the rows of the instance(image) matrix")
tf.app.flags.DEFINE_integer("image_cols", 662, "the cols of the instance(image) matrix")
tf.app.flags.DEFINE_integer("num_epochs", 500, "the total num of epochs for the training whole dataset")
tf.app.flags.DEFINE_integer("batch_size", 20, "the batch size for the data set")
tf.app.flags.DEFINE_integer("label_size", 5, "the num of identical labels")
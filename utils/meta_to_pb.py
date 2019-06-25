import tensorflow as tf
from tensorflow.summary import FileWriter

# Your .meta file
meta_path = 'model.ckpt.meta'
output_node_names = ['logits/semantic/weights']    # Output nodes

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    for v in sess.graph.get_operations():
        print(v.name)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('frozen_inference_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

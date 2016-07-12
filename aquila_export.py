"""
Export inception model given existing training checkpoints.

NOTE:
  Input images should be decoded JPEGs, of size 299 x 299 x 3 on the domain [0, 1]
"""

import os.path
import sys

import numpy as np
# This is a placeholder for a Google-internal import.

import tensorflow as tf

sys.path.append('/home/ubuntu')

from aquila.net import aquila_model

from tensorflow_serving.session_bundle import exporter


tf.app.flags.DEFINE_string('checkpoint_dir', '/data/aquila_train',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('export_dir', '/tmp/aquila_export',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")
#  THIS IS NOT YET IMPLEMENTED
tf.app.flags.DEFINE_integer('return_logits', False,
                            """Whether or not to provision for returning the logits as well.""")
FLAGS = tf.app.flags.FLAGS

MEAN_CHANNEL_VALS = [[[[92.366, 85.133, 81.674]]]]
MEAN_CHANNEL_VALS = np.array(MEAN_CHANNEL_VALS).round().astype(np.float32)

def export():
  with tf.Graph().as_default():
    # Build Aquila model.
    # Please refer to Tensorflow inception model for details.

    flat_image_size = 3 * FLAGS.image_size ** 2
    input_data = tf.placeholder(tf.uint8, shape=(None, flat_image_size))
    # reshape the images appropriately
    images = tf.reshape(input_data, (-1,
                                     FLAGS.image_size,
                                     FLAGS.image_size,
                                     3))

    # convert the images to float and subtract the channel mean. 
    images = tf.to_float(images) - MEAN_CHANNEL_VALS

    # Run inference.
    with tf.variable_scope('testtrain') as varscope:
      logits, endpoints = aquila_model.inference(images, 
        for_training=False, restore_logits=True)

    dG = tf.get_default_graph()
    print [x.name for x in dG.get_operations()]    
    # this is very annoying, but we have to do it this way to gain access to the abstract
    # features, which I didn't assign a sensible or unique name. 
    abstract_feats = dG.get_tensor_by_name('testtrain/aquila/logits/abst_feats/Relu:0')


    # Restore variables from training checkpoint.
    variable_averages = tf.train.ExponentialMovingAverage(
        aquila_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    print variables_to_restore
    saver = tf.train.Saver(variables_to_restore)
    
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
        return

      # perform the PCA
      # so the output will be 23 x 1024, we need a 1024 x 1024 matrix as the PCA. For now,
      # we will be using a "dummy" PCA where the weight matrix is just identity.
      abst_feats_pca = tf.diag(np.ones(abstract_feats.get_shape()[1].value))
      post_pca_abst_feats = tf.matmul(abstract_feats, abst_feats_pca)
      
      # Export inference model.
      model_exporter = exporter.Exporter(saver)

      signature = exporter.regression_signature(input_data, post_pca_abst_feats)
      model_exporter.init(default_graph_signature=signature)
      model_exporter.export(FLAGS.export_dir, tf.constant(global_step), sess)
      print('Successfully exported model to %s' % FLAGS.export_dir)


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()

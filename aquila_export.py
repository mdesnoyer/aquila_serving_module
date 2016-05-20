"""
Export inception model given existing training checkpoints.

NOTE:
  Input images should be decoded JPEGs, of size 299 x 299 x 3 on the domain [0, 1]
"""

import os.path
import sys

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


NUM_OUTPUTS = 1



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
    images = tf.to_float(images)

    # Run inference.
    logits, _ = aquila_model.inference(images, for_training=False, restore_logits=True)

    # Restore variables from training checkpoint.
    variable_averages = tf.train.ExponentialMovingAverage(
        aquila_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
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

      # Export inference model.
      model_exporter = exporter.Exporter(saver)
      signature = exporter.regression_signature(input_data, logits)
      model_exporter.init(default_graph_signature=signature)
      model_exporter.export(FLAGS.export_dir, tf.constant(global_step), sess)
      print('Successfully exported model to %s' % FLAGS.export_dir)


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()

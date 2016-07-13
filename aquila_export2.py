from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from datetime import datetime
from net import aquila_model as aquila
from net.slim import slim
from config import *
from sklearn.externals import joblib

import locale

pca_path = '/data/extracted/pca_new/pca.pkl'

from tensorflow_serving.session_bundle import exporter

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')

def fmt_num(num):
    '''
    accepts a number and then formats it for the locale.
    '''
    return locale.format("%d", num, grouping=True)

export_dir = '/mnt/neon/aquila-export'
pretrained_model_checkpoint_path = '/data/aquila_v2_snaps/model.ckpt-150000'

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999

sklearn_pca = joblib.load(pca_path)
C = sklearn_pca.components_
m = sklearn_pca.mean_

def inference(inputs, abs_feats=1024, for_training=True,
              restore_logits=True, scope=None,
              regularization_strength=0.000005):
    """
    Exports an inference op, along with the logits required for loss
    computation.

    :param inputs: An N x 299 x 299 x 3 sized float32 tensor (images)
    :param abs_feats: The number of abstract features to learn.
    :param for_training: Boolean, whether or not training is being performed.
    :param restore_logits: Restore the logits. This should only be done if the
    model is being trained on a previous snapshot of Aquila. If training from
    scratch, or transfer learning from inception, this should be false as the
    number of abstract features will likely change.
    :param scope: The name of the tower (i.e., GPU) this is being done on.
    :return: Logits, aux logits.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc],
            weight_decay=regularization_strength):
        with slim.arg_scope([slim.ops.conv2d],
                stddev=0.1,
                activation=tf.nn.relu,
                batch_norm_params=batch_norm_params):
            # i'm disabling batch normalization, because I'm concerned that
            # even though the authors claim it preserves representational
            # power, I don't believe their claim and I'm concerned about the
            # distortion it may introduce into the images.
            # Force all Variables to reside on the CPU.
            with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                logits, endpoints = slim.aquila.aquila(
                    inputs,
                    dropout_keep_prob=0.8,
                    num_abs_features=abs_feats,
                    is_training=for_training,
                    restore_logits=restore_logits,
                    scope=scope)
    return logits, endpoints

MEAN_CHANNEL_VALS = [[[[92.366, 85.133, 81.674]]]]
MEAN_CHANNEL_VALS = np.array(MEAN_CHANNEL_VALS).round().astype(np.float32)

flat_image_size = 3 * 299 ** 2
input_data = tf.placeholder(tf.uint8, shape=(None, flat_image_size))
# reshape the images appropriately
images = tf.reshape(input_data, (-1, 299, 299, 3))

images = tf.to_float(images) - MEAN_CHANNEL_VALS

with tf.variable_scope('testtrain') as varscope:
    logits, endpoints = inference(images, abs_feats, for_training=False,
                                  restore_logits=restore_logits,
                                  scope='testing',
                                  regularization_strength=WEIGHT_DECAY)
tf_comps = tf.constant(C.astype(np.float32))
tf_mean = tf.constant(m.astype(np.float32))
pca_feats = tf.matmul((endpoints['abstract_feats'] - tf_mean), tf_comps,
                      transpose_b=True)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
restorer = tf.train.Saver(variables_to_restore)
restorer.restore(sess, pretrained_model_checkpoint_path)
print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), pretrained_model_checkpoint_path))

model_exporter = exporter.Exporter(restorer)
signature = exporter.regression_signature(input_data, pca_feats)
model_exporter.init(default_graph_signature=signature)
global_step = 150000
model_exporter.export(export_dir, tf.constant(global_step), sess)
print('Successfully exported model to %s' % export_dir)

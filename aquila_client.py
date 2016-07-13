"""
Send JPEG image to aquila_inference server for Regression.
"""

import os
import sys
import threading

from PIL import Image
import numpy as np 

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf
from tensorflow.python.platform.logging import warn

from tensorflow_serving.aquila_serving_module import aquila_inference_pb2


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'aquila_inference service host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('image_list_file', '', 'path to a text file containing a list of images')

FLAGS = tf.app.flags.FLAGS

# validate the preprocessing method selected
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
MEAN_CHANNEL_VALS = [[[92.366, 85.133, 81.674]]]
MEAN_CHANNEL_VALS = np.array(MEAN_CHANNEL_VALS).round().astype(np.uint8)


def _prep_image(img, w=299, h=299):
  '''
  Preprocesses the image to the desired size, permitting server-side
  batching of Aquila.

  Args:
    img: A PIL image.
    w: The desired width.
    h: The desired height.
  '''
  # pad it to correct aspect ratio (16/9)
  img = _pad_to_asp(img, 16./9)
  # resize the image to 299 x 299
  img = _resize_to(img, w=299, h=299)
  return img


def _resize_to(img, w=None, h=None):
  '''
  Resizes the image to a desired width and height. If either is undefined,
  it resizes such that the defined argument is satisfied and preserves aspect
  ratio. If both are defined, resizes to satisfy both arguments without
  preserving aspect ratio.

  Args:
    img: A PIL image.
    w: The desired width.
    h: The desired height.
  '''
  ow, oh = img.size
  asp = float(ow) / oh
  if w is None and h is None:
    # do nothing
    return img
  elif w is None:
    # set the width
    w = int(h * asp)
  elif h is None:
    h = int(w / asp)
  return img.resize((w, h), Image.BILINEAR)


def _read_image(imagefn):
  '''
  This function reads in an image as a raw file and then converts
  it to a PIL image. Note that, critically, PIL must be imported before
  tensorflow for black magic reasons.

  Args:
    imagefn: A fully-qualified path to an image as a string.

  Returns:
    The PIL image requested.
  '''
  try:
    pil_image = Image.open(imagefn)
  except Exception, e:
    warn('Problem opening %s with PIL, error: %s' % (imagefn, e.message))
    return None
  try:
    # ensure that the image file is closed.
    pil_image.load()
  except Exception, e:
    warn('Problem loading %s with PIL, error: %s' % (imagefn, e.message))
    return None
  return pil_image


def _resize_to_min(img, w=None, h=None):
  '''
  Resizes an image so that its size in both dimensions is greater than or
  equal to the provided arguments. If either argument is None, that dimension
  is ignored. If the image is larger in both dimensions, then the image is
  shrunk. In either case, the aspect ratio is preserved and image size is
  minimized. If the target of interest is in the center of the frame, but the
  image has an unusual aspect ratio, center cropping is likely the best option.
  If the image has an unusual aspect ratio but is irregularly framed, padding
  the image will prevent distortion while also including the entirety of the
  original image.

  Args:
    img: A PIL image.
    w: The minimum width desired.
    h: The minimum height desired.
  '''
  ow, oh = img.size
  if w is None and h is None:
    return img
  if w is None:
    # resize to the desired height
    return _resize_to(img, h=h)
  elif h is None:
    # resize to the desired width
    return _resize_to(img, w=w)
  if ow == w and oh == h:
    # then you need not do anything
    return img
  hf = h / float(oh)  # height scale factor
  wf = w / float(ow)  # width scale factor
  if min(hf, wf) < 1.0:
    # then some scaling up is necessary. Scale up by as much as needed,
    # leaving one dimension larger than the requested amount if required.
    scale_factor = max(hf, wf)
  else:
    # scale down by the least amount to ensure both dimensions are larger
    scale_factor = min(hf, wf)
  nw = int(ow * scale_factor)
  nh = int(oh * scale_factor)
  return _resize_to(img, w=nw, h=nh)


def _center_crop_to(img, w, h):
  '''
  Center crops image to desired size. If either dimension of the image is
  already smaller than the desired dimensions, the image is not cropped.

  Args:
    img: A PIL image.
    w: The width desired.
    h: The height desired.
  '''
  ow, oh = img.size
  if ow < w or oh < h:
    return img
  upper = (oh - h) / 2
  lower = upper + h
  left = (ow - w) / 2
  right = left + w
  return img.crop((left, upper, right, lower))


def _pad_to_asp(img, asp):
  '''
  Symmetrically pads an image to have the desired aspect ratio.

  Args:
    img: A PIL image.
    asp: The aspect ratio, a float, as w / h
  '''
  ow, oh = img.size
  oasp = float(ow) / oh
  if asp > oasp:
    # the image is too narrow. Pad out width.
    nw = int(oh * asp)
    left = (nw - ow) / 2
    upper = 0
    newsize = (nw, oh)
  elif asp < oasp:
    # the image is too short. Pad out height.
    nh = int(ow / asp)
    left = 0
    upper = (nh - oh) / 2
    newsize = (ow, nh)
  else:
    return img
  nimg = np.zeros((newsize[0], newsize[1], 3)).astype(np.uint8)
  nimg += MEAN_CHANNEL_VALS  # add in the mean channel values to the padding
  nimg = Image.fromarray(nimg)
  nimg.paste(img, box=(left, upper))
  print 'Image is',nimg.size
  return nimg


def prep_aquila(image_file):
  '''
  Preprocesses an image from a fully-qualified file.
  '''
  # Load the image.
  image = _read_image(image_file)
  if image is None:
    return None
  image = _prep_image(image)
  # the images were read in with decode jpeg, which means they are
  # floats on the domain [0, 1). So let's take that into account.
  image = numpy.array(image)
  # image = image / 256.
  return image.astype(numpy.uint8)


def do_inference(hostport, concurrency, listfile):
  '''
  Performs inference over multiple images given a list of images
  as a text file, with one image per line. The image path cannot
  be relative and must be fully-qualified. Prints the results of
  the top N classes.

  Args:
    hostport: Host:port address of the mnist_inference service.
    concurrency: Maximum number of concurrent requests.
    listfile: The path to a text file containing the fully-qualified
      path to a single image per line.

  Returns:
    None.
  '''
  imagefns = []
  with open(listfile, 'r') as f:
    imagefns = f.read().splitlines()
  num_images = len(imagefns)
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = aquila_inference_pb2.beta_create_AquilaService_stub(channel)
  cv = threading.Condition()
  # this will store the ouput Aquila. We require it to map filenames
  # to their labels in the case of batching.
  inference_results = []
  result_status = {'active': 0, 'error': 0, 'done': 0}
  def done(result_future, filename):
    '''
    Callback for result_future, modifies inference_results to hold the
    output of Aquila.
    '''
    with cv:
      exception = result_future.exception()
      if exception:
        result_status['error'] += 1
        print exception
      else:
        result = result_future.result()
        inf_res = [filename, result.valence]
        inference_results.append(inf_res)
      result_status['done'] += 1
      result_status['active'] -= 1
      cv.notify()

  for imagefn in imagefns:
    image_array = prep_aquila(imagefn)
    if image_array is None:
      num_images -= 1
      continue
    request = aquila_inference_pb2.AquilaRequest()
    # this is not as efficient as i feel like it could be,
    # since you have to flatten the array then turn it into
    # a list before you extend the request image_data field.
    request.image_data = image_array.flatten().tostring()
    with cv:
      while result_status['active'] == concurrency:
        cv.wait()
      result_status['active'] += 1
    result_future = stub.Regress.future(request, 10.0)  # 10 second timeout
    result_future.add_done_callback(
        lambda result_future, filename=imagefn: done(result_future, filename))  # pylint: disable=cell-var-from-loop
  with cv:
    while result_status['done'] != num_images:
      cv.wait()
  return inference_results


def main(_):
  host, port = FLAGS.server.split(':')
  if FLAGS.image:
    # Load and preprocess the image.
    channel = implementations.insecure_channel(host, int(port))
    stub = aquila_inference_pb2.beta_create_AquilaService_stub(channel)
    image = aquila_prep(FLAGS.image)
    request = aquila_inference_pb2.AquilaRequest()
    if image is None:
      return
    request.image_data = image.extend(image_array.flatten().tolist())
    result = stub.Regress(request, 10.0)  # 10 secs timeout
    print FLAGS.image, 'Inference:', result.valence
  elif FLAGS.image_list_file:
    inference_results = do_inference(FLAGS.server,
                                     FLAGS.concurrency,
                                     FLAGS.image_list_file)
    with open('/tmp/aquila_2_test', 'a') as f:
      for filename, valence in inference_results:
        print filename, 'Inference:', valence
        cstr = [filename] + [str(x) for x in list(valence)]
        cstr = ','.join(cstr)
        f.write('%s\n' % cstr)


if __name__ == '__main__':
  tf.app.run()
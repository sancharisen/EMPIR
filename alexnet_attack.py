from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import cifar10 ## Directly load CIFAR10 dataset from KERAS
from keras.utils import np_utils

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

#from cleverhans.attacks import fgsm
from cleverhans.utils import set_log_level, parse_model_settings, build_model_save_path
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval, tf_model_load
# SANCHARI: added below for imagenet preprocessing
from examples import imagenet_preprocessing_modified# STARTED WORKING ON CBRIC GPUS, after adding __init__.py file to examples folder
from cleverhans.utils_tf import model_train_imagenet, model_train_imagenet2, model_retrain_imagenet2, model_eval_imagenet, model_eval_combinedThree_imagenet, model_eval_adv_imagenet, model_eval_CWadv_imagenet, model_eval_combinedThree_adv_imagenet, model_eval_combinedThree_CWadv_imagenet # For training and evaluating on imagenet
from cleverhans.utils_tf import model_train_imagenet_inpgrad_reg  # SANCHARI: for training with input gradient regularization
from cleverhans.utils import build_retrainedModel_save_path # For getting save paths for the retrained models
from cleverhans.utils_tf import model_eval_layer_imagenet, model_eval_layer_imagenet_singlebatch, model_eval_layer_adv_imagenet # SANCHARI: added as functions to investigate why EMPIR works better

import sys # For allowing error exits

from collections import OrderedDict # SANCHARI: for getting an ordered dict while restoring models

FLAGS = flags.FLAGS

ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_BASICITER = 4
MAX_BATCH_SIZE = 100
MAX_BATCH_SIZE = 100

# SANCHARI: copied this from cleverhans_tutorials/mnist_attack.py
# enum adversarial training types
ADVERSARIAL_TRAINING_MADRYETAL = 1
ADVERSARIAL_TRAINING_FGSM = 2
MAX_EPS = 0.3

# SANCHARI: copied this from cleverhans_tutorials/mnist_attack.py
# Scaling input to softmax
INIT_T = 1.0

#ATTACK_T = 1.0
ATTACK_T = 0.25

# _DEFAULT_IMAGE_SIZE = 227 # ALEXNET IN  CAFFE AND THE PRETRAINED MODEL TAKES A CROP OF THIS SIZE
_DEFAULT_IMAGE_SIZE = 224 # ALEXNET IN  PAPER AND DOREFANET TAKES A CROP OF THIS SIZE
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000

## Copied from resnet model
_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'Train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'Val-%05d-of-00128' % i)
        for i in range(128)]

# SANCHARI: copied the two functions below from resnet_attack.py
# Modified the parse_example_proto from resnet_attack.py to have fields according to imagenet_to_gcs_dataset.py
def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):
    image/height': _int64_feature(height),
    image/width': _int64_feature(width),
    image/colorspace': _bytes_feature(colorspace),
    image/channels': _int64_feature(channels),
    image/class/label': _int64_feature(label),
    image/class/synset': _bytes_feature(synset),
    image/format': _bytes_feature(image_format),
    image/filename': _bytes_feature(os.path.basename(filename)),
    image/encoded': _bytes_feature(image_buffer)}))

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
    'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    'image/colorspace': tf.VarLenFeature(dtype=tf.string),
    'image/channels': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/synset': tf.VarLenFeature(dtype=tf.string),
    'image/format': tf.VarLenFeature(dtype=tf.string),
    'image/filename': tf.VarLenFeature(dtype=tf.string),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string), # FLOAT or STRING ?
  } 
  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  one_hot_label = tf.one_hot(label, _NUM_CLASSES, 1, 0) #SANCHARI: convert it to a one_hot vector 

  # Directly fixing values of min and max
  xmin = tf.expand_dims([0.0], 0)
  ymin = tf.expand_dims([0.0], 0)
  xmax = tf.expand_dims([1.0], 0)
  ymax = tf.expand_dims([1.0], 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  # return features['image/encoded'], label, bbox
  return features['image/encoded'], one_hot_label, bbox

# variant of the above to parse training datasets which have labels from 1 to 1000 instead of 0 to 999
def _parse_train_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):
    image/height': _int64_feature(height),
    image/width': _int64_feature(width),
    image/colorspace': _bytes_feature(colorspace),
    image/channels': _int64_feature(channels),
    image/class/label': _int64_feature(label),
    image/class/synset': _bytes_feature(synset),
    image/format': _bytes_feature(image_format),
    image/filename': _bytes_feature(os.path.basename(filename)),
    image/encoded': _bytes_feature(image_buffer)}))

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
    'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    'image/colorspace': tf.VarLenFeature(dtype=tf.string),
    'image/channels': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/synset': tf.VarLenFeature(dtype=tf.string),
    'image/format': tf.VarLenFeature(dtype=tf.string),
    'image/filename': tf.VarLenFeature(dtype=tf.string),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string), # FLOAT or STRING ?
  } 
  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32) -1 # trying to increase accuracy of pretrained model on the training dataset
  one_hot_label = tf.one_hot(label, _NUM_CLASSES, 1, 0) #SANCHARI: convert it to a one_hot vector 

  # Directly fixing values of min and max
  xmin = tf.expand_dims([0.0], 0)
  ymin = tf.expand_dims([0.0], 0)
  xmax = tf.expand_dims([1.0], 0)
  ymax = tf.expand_dims([1.0], 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  # return features['image/encoded'], label, bbox
  return features['image/encoded'], one_hot_label, bbox

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.
  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).
  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.
  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  if is_training:
    image_buffer, label, bbox = _parse_train_example_proto(raw_record)
  else:
    image_buffer, label, bbox = _parse_example_proto(raw_record)

  # preprocess_image2 which just does decode and resize and mean subtraction
  # image = imagenet_preprocessing_modified.preprocess_image2( # for default FP model
  # image = imagenet_preprocessing_modified.preprocess_image( # For pretrained Dorefanet network but doesnt help (?) 
  image = imagenet_preprocessing_modified.preprocess_image4( # For pretrained Dorefanet network with division by standard deviation 
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
      # is_training=True)
  # SANCHARI: made is_training=True to avoid errors with image size 256
  
  image = tf.cast(image, dtype)
  # image = tf.cast(image_buffer, dtype)

  return image, label

# SANCHARI: copied function below from resnet_run_loop.py
def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1):
  """Given a Dataset with raw records, return an iterator over the records.
  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    num_parallel_batches: Number of parallel batches for tf.data.
  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size) #SANCHARI: commented as it was giving error
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Repeats the dataset for the number of epochs to train.
  # dataset = dataset.repeat(num_epochs) # SANCHARI: Not needed as reinitialization in model_train_imagenet should work

  # Parses the raw records into images and labels.
  # dataset = dataset.apply(
  #     tf.contrib.data.map_and_batch(
  #         lambda value: parse_record_fn(value, is_training, dtype),
  #         batch_size=batch_size,
  #         num_parallel_batches=num_parallel_batches,
  #         drop_remainder=False))
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dtype),
          batch_size=batch_size,
          num_parallel_batches=num_parallel_batches))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  # dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=1) 

  return dataset

def data_imagenet(nb_epochs, batch_size):
    """
    Preprocess Imagenet dataset
    :return:
    """

    # Load images from validation dataset
    # copied from resnet file in official resnet tensorflow repo
    test_dataset =tf.data.TFRecordDataset(get_filenames(is_training=False, data_dir='/home/consus/a/sen9/verifiedAI/cleverhans-attacking-bnns/imagenet_files_SS/tf_records/Val'))
    # test_dataset =tf.data.TFRecordDataset(get_filenames(is_training=True, data_dir='/home/consus/a/sen9/verifiedAI/cleverhans-attacking-bnns/imagenet_files_SS/tf_records/Train'))
    # Convert to individual records==> LEADS TO A DIFFERENT ERROR SOMEHOW
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    # test_dataset = test_dataset.apply(tf.contrib.data.parallel_interleave(
    #   tf.data.TFRecordDataset, cycle_length=10)) 
     
    train_dataset = tf.data.TFRecordDataset(get_filenames(is_training=True, data_dir='/home/consus/a/sen9/verifiedAI/cleverhans-attacking-bnns/imagenet_files_SS/tf_records/Train'))
    # Convert to individual records.==> LEADS TO A DIFFERENT ERROR SOMEHOW
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    # train_dataset = train_dataset.apply(tf.contrib.data.parallel_interleave(
    #   tf.data.TFRecordDataset, cycle_length=10)) 

    train_processed = process_record_dataset(dataset=train_dataset, is_training=True, batch_size=batch_size, shuffle_buffer=_SHUFFLE_BUFFER, num_epochs = nb_epochs, parse_record_fn=parse_record)
    
    test_processed = process_record_dataset(dataset=test_dataset, is_training=False, batch_size=batch_size, shuffle_buffer=_SHUFFLE_BUFFER, parse_record_fn=parse_record)
     
    return train_processed, test_processed 

def main(argv=None):
    model_path = FLAGS.model_path
    targeted = True if FLAGS.targeted else False
    binary = True if FLAGS.binary else False
    scale = True if FLAGS.scale else False
    #### SANCHARI: extra flags for only binary activation or weights
    binaryactv=FLAGS.binaryactv
    binaryweights=FLAGS.binaryweights
    lowprecision=FLAGS.lowprecision
    lowprecision1stLayer=FLAGS.lowprecision1stLayer
    qLinear=True if FLAGS.qLinear else False
    useSeparateSeed=True if FLAGS.useSeparateSeed else False
    seed=FLAGS.seed
    abits=FLAGS.abits
    wbits=FLAGS.wbits
    abitsList=FLAGS.abitsList
    wbitsList=FLAGS.wbitsList
    stocRound=True if FLAGS.stocRound else False
    rand=FLAGS.rand #SANCHARI: extra copied from mnist_attack.py
    ####
    learning_rate = FLAGS.learning_rate
    nb_filters = FLAGS.nb_filters
    batch_size = FLAGS.batch_size
    nb_samples = FLAGS.nb_samples
    nb_epochs = FLAGS.nb_epochs
    delay = FLAGS.delay
    eps = FLAGS.eps
    adv = FLAGS.adv
    #### SANCHARI: Extra flag to specify if a full precision and low precision combined model is used
    combined = True if FLAGS.combined else False
    attackFPmodel = True if FLAGS.attackFPmodel else False
    avg = True if FLAGS.avg else False
    model_path2 = FLAGS.model_path2
    model_path1 = FLAGS.model_path1
    weightedAvg = True if FLAGS.weightedAvg else False
    weightAlpha = FLAGS.weightAlpha
    combinedTrainable=True if FLAGS.combinedTrainable else False
    combinedTwoLP=True if FLAGS.combinedTwoLP else False
    combinedThree=True if FLAGS.combinedThree else False
    model_path3 = FLAGS.model_path3
    abits2=FLAGS.abits2
    wbits2=FLAGS.wbits2
    abits2List=FLAGS.abits2List
    wbits2List=FLAGS.wbits2List
    combinedMultiple=True if FLAGS.combinedMultiple else False
    abitsModels=FLAGS.abitsModels
    wbitsModels=FLAGS.wbitsModels
    numLPmodels=FLAGS.numLPmodels
    numFPmodels=FLAGS.numFPmodels
    LPmodelPaths=FLAGS.LPmodelPaths
    FPmodelPaths=FLAGS.FPmodelPaths
    inpgradreg = True if FLAGS.inpgradreg else False
    l2dbl = FLAGS.l2dbl
    l2cs = FLAGS.l2cs
    confusion_data = True if FLAGS.confusion_data else False
    act_sparsity = True if FLAGS.act_sparsity else False
    wt_sparsity = True if FLAGS.wt_sparsity else False
    ####

    attack = FLAGS.attack
    attack_iterations = FLAGS.attack_iterations
    #SANCHARI: extra flag for adversarial training
    nb_iter = FLAGS.nb_iter
   
    # SANCHARI: extra flag to specify if we just want to test on the pre-trained model or retrain from the saved models 
    onlyTest = True if FLAGS.onlyTest else False
    retrain = True if FLAGS.retrain else False
    retrainSavePath = FLAGS.retrainSavePath
    start_epoch = FLAGS.start_epoch

    save = False
    train_from_scratch = False

    # Imagenet specific dimensions
    img_rows = _DEFAULT_IMAGE_SIZE
    img_cols = _DEFAULT_IMAGE_SIZE
    channels = _NUM_CHANNELS
    nb_classes = _NUM_CLASSES

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    set_log_level(logging.WARNING)

    # Trying to see if features can be read: till line+24: WORKS
    # sess = tf.InteractiveSession()
    # filename = tf.train.string_input_producer(['/home/consus/a/sen9/verifiedAI/cleverhans-attacking-bnns/imagenet_files_SS/tf_records/Val/Val-00039-of-00128']) 
    # reader= tf.TFRecordReader()
    # _, serialized_example = reader.read(filename)
    # feature_map = {
    # 'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/colorspace': tf.VarLenFeature(dtype=tf.string),
    # 'image/channels': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/class/synset': tf.VarLenFeature(dtype=tf.string),
    # 'image/format': tf.VarLenFeature(dtype=tf.string),
    # 'image/filename': tf.VarLenFeature(dtype=tf.string),
    # 'image/encoded': tf.FixedLenFeature([], dtype=tf.string), # FLOAT or STRING ?
    #  }  
    # features = tf.parse_single_example(serialized_example, feature_map)
    # # Many tf.train functions use tf.train.QueueRunner,
    # # so we need to start it before we read
    # tf.train.start_queue_runners(sess)
    # 
    # # Print features
    # for name, tensor in features.items():
    #     print('{}: {}'.format(name, tensor.eval()))
    ####
    # Trying to see if features can be read: till line+27: WORKS
    # def extract_fn(data_record):
    #    features = {
    #     'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    #     'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    #     'image/colorspace': tf.VarLenFeature(dtype=tf.string),
    #     'image/channels': tf.FixedLenFeature([], dtype=tf.int64),
    #     'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    #     'image/class/synset': tf.VarLenFeature(dtype=tf.string),
    #     'image/format': tf.VarLenFeature(dtype=tf.string),
    #     'image/filename': tf.VarLenFeature(dtype=tf.string),
    #     'image/encoded': tf.FixedLenFeature([], dtype=tf.string), # FLOAT or STRING ?
    #    }
    #    sample = tf.parse_single_example(data_record, features)
    #    return sample
    # 
    # # Initialize all tfrecord paths
    # dataset = tf.data.TFRecordDataset(['/home/consus/a/sen9/verifiedAI/cleverhans-attacking-bnns/imagenet_files_SS/tf_records/Val/Val-00039-of-00128'])
    # dataset = dataset.map(extract_fn)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # with tf.Session() as sess:
    #     try:
    #         while True:
    #             data_record = sess.run(next_element)
    #             print(data_record)
    #     except:
    #         pass 
    
    # Get imagenet datasets
    
    train_dataset, test_dataset = data_imagenet(nb_epochs, batch_size)

    # Creating a one-shot iterators
    # train_iterator = train_dataset.make_one_shot_iterator()
    # test_iterator = test_dataset.make_one_shot_iterator()
    # Creating a initializable iterators
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # Getting next elements from the iterators
    next_test_element = test_iterator.get_next()
    next_train_element = train_iterator.get_next()

    # for i in range(100):
    #     print("i is %d" %(i))  
    #     value = sess.run(next_train_element)
    # i = 0
    # with tf.Session() as sess2:
    #     try:
    #         while True:
    #             print("i is %d" %(i)) # Went till i is 5, ERROR: OP_REQUIRES failed at iterator_ops.cc:891 : Invalid argument: Expected begin[0] in [0, 255], but got -1
    #             data_record = sess2.run(next_test_element)
    #             # print(data_record)
    #             i = i+1
    #     except:
    #         pass
    
    train_x, train_y = train_iterator.get_next()
    test_x, test_y = test_iterator.get_next()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    # y = tf.placeholder(tf.float32, shape=(None)) #SANCHARI: Changing placeholder to 1D 
    phase = tf.placeholder(tf.bool, name="phase")

    # SANCHARI: copied from cleverhans_tutorials/mnist_attack.py for attempting to break unscaled network.
    logits_scalar = tf.placeholder_with_default(
        INIT_T, shape=(), name="logits_temperature")
   
    if combinedMultiple:
        if (FPmodelPaths is None) and (numFPmodels==0): # only LP models
            if (len(LPmodelPaths) != numLPmodels):
                train_from_scratch = True
            else:
                train_from_scratch = False
        elif (len(LPmodelPaths) != numLPmodels) or (len(FPmodelPaths) != numFPmodels):
            train_from_scratch = True
        else:
            train_from_scratch = False
    elif combinedThree: 
        if (model_path1 is None or model_path2 is None or model_path3 is None):
            train_from_scratch = True
        else:
            train_from_scratch = False
    elif combined or combinedTrainable or combinedTwoLP: 
        if (model_path1 is None or model_path2 is None):
            train_from_scratch = True
        else:
            train_from_scratch = False
    #### default below 
    # if model_path is not None:
    elif model_path is not None:
        if os.path.exists(model_path):
            # check for existing model in immediate subfolder
            if any(f.endswith('.meta') for f in os.listdir(model_path)):
                train_from_scratch = False
                if retrain: # Check if we have to retrain from the existing model
                    if retrainSavePath is None:
                        print("Error: Provide path to save the retrained model using --retrainSavePath")
                        sys.exit(1)
                    retrainSavePath = build_retrainedModel_save_path(
                    retrainSavePath, binary, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay, scale)
                    save = True
                else:   
                    # binary, scale, nb_filters, batch_size, learning_rate, nb_epochs, adv = parse_model_settings(
                    #     model_path) # default
                    binary, scale, nb_filters, _, learning_rate, nb_epochs, adv = parse_model_settings(
                        model_path) # dont restore the batch_size as it causes issues with CW attack

            else:
                model_path = build_model_save_path(
                    model_path, binary, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay, scale)
                print(model_path)
                save = True
                train_from_scratch = True
    else:
        if onlyTest:
            train_from_scratch = False
        else:
            train_from_scratch = True  # train from scratch, but don't save since no path given
    
    if combinedMultiple: # For a version with multiple combined models 
       if (abitsModels is None) or (abitsModels is None): # Layer wise separate quantization not specified for first model
           print("Error: the number of bits for constant precision weights and activations across layers for the first model have to specified using wbitsModels and abitsModels flags")
           sys.exit(1)
       if (len(abitsModels) != numLPmodels) or (len(wbitsModels) != numLPmodels):
           print("Error: Need to specify the precisions for activations and weights for all the low precision models")  
           sys.exit(1)
       
       if (train_from_scratch):
           print ("The combinedMultiple model cannot be trained from scratch")
           sys.exit(1)
       from cleverhans_tutorials.tutorial_models import make_multiple_combined_alexnet
       model = make_multiple_combined_alexnet(phase, logits_scalar, 'lp', 'fp_', wbitsModels, abitsModels, numLPmodels, numFPmodels, input_shape=(
        None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useLPbias=False, useBatchNorm=True, onlyTest=onlyTest) # SANCHARI: commented above and copied this from cleverhans_tutorials/mnist_attack.py
    
    elif combinedTrainable: # For trainable version of combined model with both high and low precision
       if (abits==0) or (wbits==0):
           print("Error: the number of bits for activations and weights have to specified using abits and wbits flags")
           sys.exit(1)
       if (train_from_scratch):
           print ("The combined model cannot be trained from scratch")
           sys.exit(1)
       
       from cleverhans_tutorials.tutorial_models import make_trainable_combined_alexnet
       model = make_trainable_combined_alexnet(
           phase, logits_scalar, 'lp_', 'fp_', wbits, abits, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, avg=avg, weightedAvg=weightedAvg, alpha=weightAlpha) 
    elif combinedTwoLP: # For trainable version of combined model with both high and low precision
       if (abits==0) or (wbits==0) or (abits2==0) or (wbits2==0):
           print("Error: the number of bits for activations and weights for both the LP models have to specified using abits and wbits, abits2 and wbits2 flags")
           sys.exit(1)
       if (train_from_scratch):
           print ("The combined model cannot be trained from scratch")
           sys.exit(1)
       from cleverhans_tutorials.tutorial_models import make_LPcombined_alexnet
       model = make_LPcombined_alexnet(
           phase, logits_scalar, 'lp1_', 'lp2_', wbits, abits, wbits2, abits2, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, avg=avg, weightedAvg=weightedAvg, alpha=weightAlpha) 
    elif combinedThree: # For trainable version of combined model with both high and low precision
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified for first model
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the first model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec1 = 1
       elif (len(wbitsList) != 6) or (len(abitsList) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer of the first model")  
           sys.exit(1)
       else: 
           fixedPrec1 = 0
       
       if (wbits2List is None) or (abits2List is None): # Layer wise separate quantization not specified for second model
           if (wbits2==0) or (abits2==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the second model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec2 = 1
       elif (len(wbits2List) != 6) or (len(abits2List) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer of the second model")  
           sys.exit(1)
       else: 
           fixedPrec2 = 0

       if (fixedPrec2 != 1) or (fixedPrec1 != 1): # Atleast one of the models have separate precisions per layer
           fixedPrec=0
           print("Within atleast one model has separate precisions")
           if (fixedPrec1 == 1): # first layer has fixed precision
               abitsList = (abits, abits, abits, abits, abits, abits)
               wbitsList = (wbits, wbits, wbits, wbits, wbits, wbits)
           if (fixedPrec2 == 1): # second layer has fixed precision
               abits2List = (abits2, abits2, abits2, abits2, abits2, abits2)
               wbits2List = (wbits2, wbits2, wbits2, wbits2, wbits2, wbits2)
       else:
           fixedPrec=1
       
       if (train_from_scratch):
           print ("The combinedThree model cannot be trained from scratch")
           sys.exit(1)
       if fixedPrec == 1:
           from cleverhans_tutorials.tutorial_models import make_three_combined_alexnet
           model = make_three_combined_alexnet(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbits, abits, wbits2, abits2, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, avg=avg, weightedAvg=weightedAvg, alpha=weightAlpha) 
       else:
           from cleverhans_tutorials.tutorial_models import make_layerwise_three_combined_alexnet
           model = make_layerwise_three_combined_alexnet(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbitsList, abitsList, wbits2List, abits2List, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, avg=avg, weightedAvg=weightedAvg, alpha=weightAlpha) 
    elif lowprecision1stLayer: # low precision variant that also quantizes the 1st layer
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers have to specified using wbits and abits flags")
               sys.exit(1)
           else:
               fixedPrec = 1
       elif (len(wbitsList) != 6) or (len(abitsList) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer")  
           sys.exit(1)
       else: 
           fixedPrec = 0
       
       # if fixedPrec:
           
       ### For training from scratch
       from cleverhans_tutorials.tutorial_models import make_basic_lowprecision_alexnet
       model = make_basic_lowprecision_alexnet(phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
        None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useSeparateSeed=useSeparateSeed, 
        seed=seed, useLPbias=False, useBatchNorm=True, onlyTest=onlyTest, quantize1stConv=True) # SANCHARI: LP vairant resembling alexnet-dorefa.py 
    elif lowprecision:
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers have to specified using wbits and abits flags")
               sys.exit(1)
           else:
               fixedPrec = 1
       elif (len(wbitsList) != 6) or (len(abitsList) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer")  
           sys.exit(1)
       else: 
           fixedPrec = 0
       
       if fixedPrec:
           
           ### For training from scratch
           from cleverhans_tutorials.tutorial_models import make_basic_lowprecision_alexnet
           model = make_basic_lowprecision_alexnet(phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useSeparateSeed=useSeparateSeed, seed=seed, useLPbias=False, useBatchNorm=True, onlyTest=onlyTest) # SANCHARI: LP vairant resembling alexnet-dorefa.py 
           
           ### For training from pretrained LP models
           # from cleverhans_tutorials.tutorial_models import make_basic_lowprecision_alexnet_pretrained
           # model = make_basic_lowprecision_alexnet_pretrained(phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
           #  None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useSeparateSeed=useSeparateSeed, seed=seed, onlyTest=onlyTest) # SANCHARI: LP vairant with bias, gives exact FP accuracy for LP with 32 bits for pretrained models 
           
           # model = make_basic_lowprecision_alexnet_pretrained(phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
           #  None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useSeparateSeed=useSeparateSeed, seed=seed, useLPbias=False, onlyTest=onlyTest) # SANCHARI: LP variant without bias 
           
           # model = make_basic_lowprecision_alexnet_pretrained(phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
           #  None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useSeparateSeed=useSeparateSeed, seed=seed, useLPbias=False, useBatchNorm=True, onlyTest=onlyTest) # SANCHARI: LP vairant without bias and with BatchNorm, for 8 bit pre trained models
       else:
           from cleverhans_tutorials.tutorial_models import make_layerwise_lowprecision_alexnet
           model = make_layerwise_lowprecision_alexnet(phase, logits_scalar, 'lp_', wbitsList, abitsList, input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes, useSeparateSeed=useSeparateSeed, seed=seed, useLPbias=False, useBatchNorm=True, onlyTest=onlyTest) # SANCHARI: commented above and copied this from cleverhans_tutorials/mnist_attack.py
    else:
        if onlyTest: 
            ### For training from pretrained model
            from cleverhans_tutorials.tutorial_models import make_basic_alexnet
            model = make_basic_alexnet(phase, logits_scalar, 'fp_', input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes) # SANCHARI: commented above and copied this from cleverhans_tutorials/mnist_attack.py
        else:
            ### For training from scratch
            from cleverhans_tutorials.tutorial_models import make_basic_alexnet_from_scratch
            model = make_basic_alexnet_from_scratch(phase, logits_scalar, 'fp_', input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes) # SANCHARI: commented above and copied this from cleverhans_tutorials/mnist_attack.py

    # SANCHARI:separate calling function for model in case of combinedTrainable
    if combinedTrainable or combinedTwoLP or combinedThree or combinedMultiple:
        preds = model.combined_call(x, reuse=False)
    # SANCHARI: two predictions for model1 and model2 in case of combined
    elif combined:
        preds1 = model1(x, reuse=False)
        preds2 = model2(x, reuse=False)
    else:
    ##default
        preds = model(x, reuse=False)
    print("Defined TensorFlow model graph.")

    rng = np.random.RandomState([2017, 8, 30])

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        ##SANCHARI: separate evaluation function for combined model and combinedThree model
        if combinedThree or combinedMultiple:
            acc = model_eval_combinedThree_imagenet(
                sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, args=eval_params)
        ##SANCHARI: separate evaluation function for combined model
        elif combined:
            if avg:
                acc = model_eval_combined_avg(
                    sess, x, y, preds1, preds2, X_test, Y_test, phase=phase, args=eval_params)
            else:
                acc = model_eval_combined(
                    sess, x, y, preds1, preds2, X_test, Y_test, phase=phase, args=eval_params)
        else: #default below
            acc = model_eval_imagenet(
                sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an Imagenet model
    train_params = {
        'binary': binary,
        'lowprecision': lowprecision,
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_name': 'train loss',
        'filename': 'model',
        'reuse_global_step': False,
        'train_scope': 'train',
        'is_training': True
    }

    #SANCHARI  copied this from mnist_attack, default before
    if adv != 0:
        if adv == ADVERSARIAL_TRAINING_MADRYETAL:
            from cleverhans.attacks import MadryEtAl
            train_attack_params = {'eps': MAX_EPS, 'eps_iter': 0.01,
                                   'nb_iter': nb_iter}
            train_attacker = MadryEtAl(model, sess=sess)

        elif adv == ADVERSARIAL_TRAINING_FGSM:
            from cleverhans.attacks import FastGradientMethod
            stddev = int(np.ceil((MAX_EPS * 255) // 2))
            train_attack_params = {'eps': tf.abs(tf.truncated_normal(
                shape=(batch_size, 1, 1, 1), mean=0, stddev=stddev))}
            train_attacker = FastGradientMethod(model, back='tf', sess=sess)
        # create the adversarial trainer
        train_attack_params.update({'clip_min': 0., 'clip_max': 1.})
        adv_x_train = train_attacker.generate(x, phase, **train_attack_params)
        preds_adv_train = model.get_probs(adv_x_train)

        eval_attack_params = {'eps': MAX_EPS, 'clip_min': 0., 'clip_max': 1.}
        adv_x_eval = train_attacker.generate(x, phase, **eval_attack_params)
        preds_adv_eval = model.get_probs(adv_x_eval)  # * logits_scalar
   #  if adv:
   #      from cleverhans.attacks import FastGradientMethod
   #      fgsm = FastGradientMethod(model, back='tf', sess=sess)
   #      fgsm_params = {'eps': eps, 'clip_min': 0., 'clip_max': 1.}
   #      adv_x_train = fgsm.generate(x, phase, **fgsm_params)
   #      preds_adv = model.get_probs(adv_x_train)

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

        # do clean training for 'nb_epochs' or 'delay' epochs with constant learning rate
        # model_train_imagenet(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
        #             evaluate=evaluate, args=train_params, save=save, rng=rng)
        
        if inpgradreg: 
           model_train_imagenet_inpgrad_reg(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                    evaluate=evaluate, l2dbl = l2dbl, l2cs = l2cs, args=train_params, save=save, rng=rng)
        else:
            # do clean training for 'nb_epochs' or 'delay' epochs with learning rate reducing with time
            model_train_imagenet2(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                    evaluate=evaluate, args=train_params, save=save, rng=rng)

        # optionally do additional adversarial training
        if adv:
            print("Adversarial training for %d epochs" % (nb_epochs - delay))
            train_params.update({'nb_epochs': nb_epochs - delay})
            train_params.update({'reuse_global_step': True})
            # SANCHARI: changed the predictions argument from default below
            model_train_imagenet2(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                    predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params, save=save, rng=rng)
    else:
        if combined or combinedTrainable: ## Combined models have to loaded from different paths
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            # First 11 variables from path1
            stored_variables = ['lp_conv1_init/k', 'lp_conv1_init/b', 'lp_conv2_init/k', 'lp_conv3_init/k', 'lp_conv4_init/k', 'lp_conv5_init/k', 'lp_ip1init/W', 'lp_ip1init/b', 'lp_ip2init/W', 'lp_logits_init/W', 'lp_logits_init/b']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[:11]))) # dict was messing with the order 
            # Restore the first set of variables from model_path1
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Second 16 variables from path2
            stored_variables = ['fp_conv1_init/k', 'fp_conv1_init/b', 'fp_conv2_init/k', 'fp_conv2_init/b', 'fp_conv3_init/k', 'fp_conv3_init/b', 'fp_conv4_init/k', 'fp_conv4_init/b', 'fp_conv5_init/k', 'fp_conv5_init/b', 'fp_ip1init/W', 'fp_ip1init/b', 'fp_ip2init/W', 'fp_ip2init/b', 'fp_logits_init/W', 'fp_logits_init/b']

            variable_dict = dict(OrderedDict(zip(stored_variables, variables[11:27])))
            saver2 = tf.train.Saver(variable_dict)
            saver2.restore(sess, tf.train.latest_checkpoint(model_path2))
            # Next 24 batch norm variables from path1
            stored_variables = ['lp__batchNorm1/batch_normalization/gamma', 'lp__batchNorm1/batch_normalization/beta', 'lp__batchNorm1/batch_normalization/moving_mean', 'lp__batchNorm1/batch_normalization/moving_variance', 'lp__batchNorm2/batch_normalization/gamma', 'lp__batchNorm2/batch_normalization/beta', 'lp__batchNorm2/batch_normalization/moving_mean', 'lp__batchNorm2/batch_normalization/moving_variance', 'lp__batchNorm3/batch_normalization/gamma', 'lp__batchNorm3/batch_normalization/beta', 'lp__batchNorm3/batch_normalization/moving_mean', 'lp__batchNorm3/batch_normalization/moving_variance', 'lp__batchNorm4/batch_normalization/gamma', 'lp__batchNorm4/batch_normalization/beta', 'lp__batchNorm4/batch_normalization/moving_mean', 'lp__batchNorm4/batch_normalization/moving_variance', 'lp__batchNorm5/batch_normalization/gamma', 'lp__batchNorm5/batch_normalization/beta', 'lp__batchNorm5/batch_normalization/moving_mean', 'lp__batchNorm5/batch_normalization/moving_variance', 'lp__batchNorm6/batch_normalization/gamma', 'lp__batchNorm6/batch_normalization/beta', 'lp__batchNorm6/batch_normalization/moving_mean', 'lp__batchNorm6/batch_normalization/moving_variance']

            variable_dict = dict(OrderedDict(zip(stored_variables, variables[27:51])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            print('Restored model from %s and %s' %(model_path1, model_path2))
        elif combinedTwoLP:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # First 11 variables from path1
            stored_variables = ['lp_conv1_init/k', 'lp_conv1_init/b', 'lp_conv2_init/k', 'lp_conv3_init/k', 'lp_conv4_init/k', 'lp_conv5_init/k', 'lp_ip1init/W', 'lp_ip1init/b', 'lp_ip2init/W', 'lp_logits_init/W', 'lp_logits_init/b']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[:11]))) # only dict was messing with the order
            # Restore the first set of variables from model_path1
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Restore the second set of variables from model_path2
            # Second 11 variables from path2
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[11:22])))
            saver2 = tf.train.Saver(variable_dict)
            saver2.restore(sess, tf.train.latest_checkpoint(model_path2))
            # Next 24 batch norm variables from path1
            stored_variables = ['lp__batchNorm1/batch_normalization/gamma', 'lp__batchNorm1/batch_normalization/beta', 'lp__batchNorm1/batch_normalization/moving_mean', 'lp__batchNorm1/batch_normalization/moving_variance', 'lp__batchNorm2/batch_normalization/gamma', 'lp__batchNorm2/batch_normalization/beta', 'lp__batchNorm2/batch_normalization/moving_mean', 'lp__batchNorm2/batch_normalization/moving_variance', 'lp__batchNorm3/batch_normalization/gamma', 'lp__batchNorm3/batch_normalization/beta', 'lp__batchNorm3/batch_normalization/moving_mean', 'lp__batchNorm3/batch_normalization/moving_variance', 'lp__batchNorm4/batch_normalization/gamma', 'lp__batchNorm4/batch_normalization/beta', 'lp__batchNorm4/batch_normalization/moving_mean', 'lp__batchNorm4/batch_normalization/moving_variance', 'lp__batchNorm5/batch_normalization/gamma', 'lp__batchNorm5/batch_normalization/beta', 'lp__batchNorm5/batch_normalization/moving_mean', 'lp__batchNorm5/batch_normalization/moving_variance', 'lp__batchNorm6/batch_normalization/gamma', 'lp__batchNorm6/batch_normalization/beta', 'lp__batchNorm6/batch_normalization/moving_mean', 'lp__batchNorm6/batch_normalization/moving_variance']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[22:46])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Final 24 batch norm variables from path1
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[46:70])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path2))
        elif combinedThree: ## CombinedThree models have to loaded from different paths
            # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print(i)   # print all variables in a graph
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            # First 11 variables from path1
            stored_variables = ['lp_conv1_init/k', 'lp_conv1_init/b', 'lp_conv2_init/k', 'lp_conv3_init/k', 'lp_conv4_init/k', 'lp_conv5_init/k', 'lp_ip1init/W', 'lp_ip1init/b', 'lp_ip2init/W', 'lp_logits_init/W', 'lp_logits_init/b']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[:11]))) # only dict was messing with the order
            # Restore the first set of variables from model_path1
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Restore the second set of variables from model_path2
            # Second 11 variables from path2
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[11:22])))
            saver2 = tf.train.Saver(variable_dict)
            saver2.restore(sess, tf.train.latest_checkpoint(model_path2))
            # Third 16 variables from path3
            stored_variables = ['fp_conv1_init/k', 'fp_conv1_init/b', 'fp_conv2_init/k', 'fp_conv2_init/b', 'fp_conv3_init/k', 'fp_conv3_init/b', 'fp_conv4_init/k', 'fp_conv4_init/b', 'fp_conv5_init/k', 'fp_conv5_init/b', 'fp_ip1init/W', 'fp_ip1init/b', 'fp_ip2init/W', 'fp_ip2init/b', 'fp_logits_init/W', 'fp_logits_init/b']

            variable_dict = dict(OrderedDict(zip(stored_variables, variables[22:38])))
            saver3 = tf.train.Saver(variable_dict)
            saver3.restore(sess, tf.train.latest_checkpoint(model_path3))
            # Next 24 batch norm variables from path1
            stored_variables = ['lp__batchNorm1/batch_normalization/gamma', 'lp__batchNorm1/batch_normalization/beta', 'lp__batchNorm1/batch_normalization/moving_mean', 'lp__batchNorm1/batch_normalization/moving_variance', 'lp__batchNorm2/batch_normalization/gamma', 'lp__batchNorm2/batch_normalization/beta', 'lp__batchNorm2/batch_normalization/moving_mean', 'lp__batchNorm2/batch_normalization/moving_variance', 'lp__batchNorm3/batch_normalization/gamma', 'lp__batchNorm3/batch_normalization/beta', 'lp__batchNorm3/batch_normalization/moving_mean', 'lp__batchNorm3/batch_normalization/moving_variance', 'lp__batchNorm4/batch_normalization/gamma', 'lp__batchNorm4/batch_normalization/beta', 'lp__batchNorm4/batch_normalization/moving_mean', 'lp__batchNorm4/batch_normalization/moving_variance', 'lp__batchNorm5/batch_normalization/gamma', 'lp__batchNorm5/batch_normalization/beta', 'lp__batchNorm5/batch_normalization/moving_mean', 'lp__batchNorm5/batch_normalization/moving_variance', 'lp__batchNorm6/batch_normalization/gamma', 'lp__batchNorm6/batch_normalization/beta', 'lp__batchNorm6/batch_normalization/moving_mean', 'lp__batchNorm6/batch_normalization/moving_variance']

            variable_dict = dict(OrderedDict(zip(stored_variables, variables[38:62])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Final 24 batch norm variables from path1
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[62:86])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path2))
        elif combinedMultiple: ## CombinedMultiple models have to loaded from different paths
            # print("variables in the graph")
            # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print(i)   # print all variables in a graph
            stored_variables = ['lp_conv1_init/k', 'lp_conv1_init/b', 'lp_conv2_init/k', 'lp_conv3_init/k', 'lp_conv4_init/k', 'lp_conv5_init/k', 'lp_ip1init/W', 'lp_ip1init/b', 'lp_ip2init/W', 'lp_logits_init/W', 'lp_logits_init/b']
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # variable_dict = dict(zip(variables[:4], stored_variables)) # Error with size, Dimensions must be equal, but are 128 and 64 for 'Less' (op: 'Less') with input shapes: [128,10], [6,6,64,128] line saver = tf.train.Saver(variable_dict) error
            for i in range(numLPmodels):
                variable_dict = dict(OrderedDict(zip(stored_variables, variables[i*11:(i+1)*11])))
                # print("variable_dict is ", variable_dict) 
                # Restore the first set of variables from model_path1
                saver = tf.train.Saver(variable_dict)
                saver.restore(sess, tf.train.latest_checkpoint(LPmodelPaths[i]))

            # Restore the second set of variables from model_path2
            stored_variables = ['fp_conv1_init/k', 'fp_conv1_init/b', 'fp_conv2_init/k', 'fp_conv2_init/b', 'fp_conv3_init/k', 'fp_conv3_init/b', 'fp_conv4_init/k', 'fp_conv4_init/b', 'fp_conv5_init/k', 'fp_conv5_init/b', 'fp_ip1init/W', 'fp_ip1init/b', 'fp_ip2init/W', 'fp_ip2init/b', 'fp_logits_init/W', 'fp_logits_init/b']

            for i in range(numFPmodels):
                variable_dict = dict(OrderedDict(zip(stored_variables, variables[(numLPmodels*11 + i*16):(numLPmodels*11 + (i+1)*16)]))) 
                # Restore the first set of variables from model_path1
                saver = tf.train.Saver(variable_dict)
                saver.restore(sess, tf.train.latest_checkpoint(FPmodelPaths[i]))

            # Next 24 batch norm variables from each of the LP paths
            stored_variables = ['lp__batchNorm1/batch_normalization/gamma', 'lp__batchNorm1/batch_normalization/beta', 'lp__batchNorm1/batch_normalization/moving_mean', 'lp__batchNorm1/batch_normalization/moving_variance', 'lp__batchNorm2/batch_normalization/gamma', 'lp__batchNorm2/batch_normalization/beta', 'lp__batchNorm2/batch_normalization/moving_mean', 'lp__batchNorm2/batch_normalization/moving_variance', 'lp__batchNorm3/batch_normalization/gamma', 'lp__batchNorm3/batch_normalization/beta', 'lp__batchNorm3/batch_normalization/moving_mean', 'lp__batchNorm3/batch_normalization/moving_variance', 'lp__batchNorm4/batch_normalization/gamma', 'lp__batchNorm4/batch_normalization/beta', 'lp__batchNorm4/batch_normalization/moving_mean', 'lp__batchNorm4/batch_normalization/moving_variance', 'lp__batchNorm5/batch_normalization/gamma', 'lp__batchNorm5/batch_normalization/beta', 'lp__batchNorm5/batch_normalization/moving_mean', 'lp__batchNorm5/batch_normalization/moving_variance', 'lp__batchNorm6/batch_normalization/gamma', 'lp__batchNorm6/batch_normalization/beta', 'lp__batchNorm6/batch_normalization/moving_mean', 'lp__batchNorm6/batch_normalization/moving_variance']
            for i in range(numLPmodels):
                variable_dict = dict(OrderedDict(zip(stored_variables, variables[(numLPmodels*11 + numFPmodels*16 + i*24) : (numLPmodels*11  + numFPmodels*16 + (i+1)*24)]))) 
                # print("variable_dict is ", variable_dict) 
                # Restore the first set of variables from model_path1
                saver = tf.train.Saver(variable_dict)
                saver.restore(sess, tf.train.latest_checkpoint(LPmodelPaths[i]))
        elif onlyTest:
        # Initialize all variables required if no training is done and we want to do only test on the downloaded model
            sess.run(tf.global_variables_initializer())
        elif retrain: # restoring the model trained using this setup and retrain on it, not a downloaded one
            tf_model_load(sess, model_path)
            train_params.update({'log_dir': retrainSavePath}) # update the save path

            print('Restored model from %s and retraining in another path %s' %(model_path, retrainSavePath))
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

            # do clean training for 'nb_epochs' or 'delay' epochs with constant learning rate
            # model_train_imagenet(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
            #             evaluate=evaluate, args=train_params, save=save, rng=rng)
            
            # do clean training for 'nb_epochs' or 'delay' epochs with learning rate reducing with time
            model_retrain_imagenet2(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                        evaluate=evaluate, args=train_params, save=save, rng=rng, start_epoch=start_epoch)

            # optionally do additional adversarial training
            if adv:
                print("Adversarial training for %d epochs" % (nb_epochs - delay))
                train_params.update({'nb_epochs': nb_epochs - delay})
                train_params.update({'reuse_global_step': True})
                # SANCHARI: changed the predictions argument from default below
                model_train_imagenet2(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                        predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params, save=save, rng=rng)
                evaluate()
        #default below
        else: # restoring the model trained using this setup, not a downloaded one
            tf_model_load(sess, model_path)
            print('Restored model from %s' % model_path)
            # evaluate()


    # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    if combinedThree or combinedMultiple: ## SANCHARI: CombinedThree models have to be evaluated with a separate function
        # preds = tf.Print(preds, ["preds[0] in alexnet_attack", preds[0]])
        accuracy = model_eval_combinedThree_imagenet(sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, feed={phase: False}, args=eval_params)
    elif combined: ## SANCHARI: Combined models have to be evaluated with a separate function
        if avg:
            accuracy = model_eval_combined_avg(sess, x, y, preds1, preds2, X_test, Y_test, phase=phase, feed={phase: False}, args=eval_params)
        else:
            accuracy = model_eval_combined(sess, x, y, preds1, preds2, X_test, Y_test, phase=phase, feed={phase: False}, args=eval_params)
    else: #default below
        accuracy = model_eval_imagenet(sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, feed={phase: False}, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Build dataset
    ###########################################################################

    adv_inputs = test_x #adversarial inputs can be generated from any of the test examples 

    ###########################################################################
    # Craft adversarial examples using generic approach
    ###########################################################################
    # att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
    if attack == ATTACK_CARLINI_WAGNER_L2:
        att_batch_size = np.minimum(nb_samples, batch_size) #SANCHARI: corrected this to have current batch size instead of MAX_BATCH_SIZE to prevent errors in CW attack
    else:
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
    nb_adv_per_sample = 1
    adv_ys = None
    yname = "y"

    print('Crafting ' + str(nb_samples) + ' * ' + str(nb_adv_per_sample) +
          ' adversarial examples')
    print("This could take some time ...")

    #SANCHARI: define model type to ensure targeted attack generation uses correct labels 
    # Setting all model_type to default should absolutely give the ECML results
    if combinedTrainable:
        model_type = 'combinedTwo'
    elif combinedMultiple:
        model_type = 'combinedMultiple'
    elif combinedThree:
        model_type = 'combinedThree'
    else:
        model_type = 'default'

    # avgType = 'all' # For averaging across all models
    avgType = 'correct' # For ICLR submission results

    if attack == ATTACK_CARLINI_WAGNER_L2:
        from cleverhans.attacks import CarliniWagnerL2
        if combined: #SANCHARI: for the combined model, define attack on the fp model if attackFPmodel is set
            if attackFPmodel:
                attacker = CarliniWagnerL2(model2, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
            else:
                attacker = CarliniWagnerL2(model1, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        else: #default below
            attacker = CarliniWagnerL2(model, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.1,
                         'batch_size': att_batch_size,
                         'initial_const': 10,
                         }
    elif attack == ATTACK_JSMA:
        from cleverhans.attacks import SaliencyMapMethod
        if combined: #SANCHARI: for the combined model, define attack on the fp model if attackFPmodel is set
            if attackFPmodel:
                attacker = SaliencyMapMethod(model2, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
            else:
                attacker = SaliencyMapMethod(model1, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        else: #default below
            attacker = SaliencyMapMethod(model, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        attack_params = {'theta': 1., 'gamma': 0.1}
    elif attack == ATTACK_FGSM:
        from cleverhans.attacks import FastGradientMethod
        if combined: #SANCHARI: for the combined model, define attack on the fp model if attackFPmodel is set
            if attackFPmodel:
                attacker = FastGradientMethod(model2, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
            else:
                attacker = FastGradientMethod(model1, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        else: #default below
            attacker = FastGradientMethod(model, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        attack_params = {'eps': eps}
    elif attack == ATTACK_MADRYETAL:
        from cleverhans.attacks import MadryEtAl
        if combined: #SANCHARI: for the combined model, define attack on the fp model if attackFPmodel is set
            if attackFPmodel:
                attacker = MadryEtAl(model2, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
            else:
                attacker = MadryEtAl(model1, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        else: #default below
            attacker = MadryEtAl(model, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    elif attack == ATTACK_BASICITER:
        print('Attack: BasicIterativeMethod')
        from cleverhans.attacks import BasicIterativeMethod
        if combined: #SANCHARI: for the combined model, define attack on the fp model if attackFPmodel is set
            if attackFPmodel:
                attacker = BasicIterativeMethod(model2, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
            else:
                attacker = BasicIterativeMethod(model1, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        else: #default below
            attacker = BasicIterativeMethod(model, back='tf', sess=sess, model_type=model_type, avgType=avgType, num_classes=nb_classes)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    else:
        print("Attack undefined")
        sys.exit(1)

    # attack_params.update({'clip_min': 0., 'clip_max': 1.})
    attack_params.update({'clip_min': -2.2, 'clip_max': 2.7}) # Since max and min for imagenet turns out to be around -2.11 and 2.12, gives exact accuracy with eps=0
    eval_params = {'batch_size': att_batch_size}
    '''
    adv_x = attacker.generate(x, phase, **attack_params)
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    eval_params = {'batch_size': att_batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [adv_inputs], feed={
                             phase: False}, args=eval_params)
    '''

    # assert X_test_adv.shape[0] == nb_samples, X_test_adv.shape #SANCHARI: commented this out as mnist_attack.py doesnt have this as well
    # Evaluate the accuracy of the alexnet model on adversarial examples
    ##SANCHARI: separate evaluation function for combined model
    print("Evaluating un-targeted results")
    if combinedThree or combinedMultiple:
        if attack == ATTACK_CARLINI_WAGNER_L2: #special function for CW attack
            adv_accuracy = model_eval_combinedThree_CWadv_imagenet(sess, x, y, preds, test_iterator, 
                            test_x, test_y, phase=phase, args=eval_params, nb_samples=nb_samples, attacker=attacker, attack_params=attack_params)
        else:
            adv_accuracy = model_eval_combinedThree_adv_imagenet(sess, x, y, preds, test_iterator, 
                            test_x, test_y, phase=phase, args=eval_params, nb_samples=nb_samples, attacker=attacker, attack_params=attack_params)
    elif combined:
        if avg:
            adv_accuracy = model_eval_combined_avg(sess, x, y, preds1, preds2, X_test_adv, Y_test[:nb_samples], phase=phase, args=eval_params)
        else:
            adv_accuracy = model_eval_combined(sess, x, y, preds1, preds2, X_test_adv, Y_test[:nb_samples], phase=phase, args=eval_params)
    else: #default below: advesaries generated within the eval function as well
        if attack == ATTACK_CARLINI_WAGNER_L2: #special function for CW attack
            adv_accuracy = model_eval_CWadv_imagenet(sess, x, y, preds, test_iterator, 
                            test_x, test_y, phase=phase, args=eval_params, nb_samples=nb_samples, attacker=attacker, attack_params=attack_params)
        else:
            adv_accuracy = model_eval_adv_imagenet(sess, x, y, preds, test_iterator, 
                            test_x, test_y, phase=phase, args=eval_params, nb_samples=nb_samples, attacker=attacker, attack_params=attack_params)
    
    # Compute the number of adversarial examples that were successfully found
    print('Test accuracy on adversarial examples {0:.4f}'.format(adv_accuracy))


    ####### Generating data for confusion matrix
    if confusion_data:
        import matplotlib.pyplot as plt
        pred_val, Y_val = model_eval_layer_imagenet(sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, args=eval_params)
        if not combinedThree and not combinedMultiple:
            pred_class = np.argmax(pred_val, axis=-1)  
        else:
            pred_class = pred_val
        actual_class = np.argmax(Y_val, axis=-1)
        
        confusion = np.zeros([nb_classes, nb_classes])
        for i in range(len(actual_class)):
            confusion[actual_class[i]][pred_class[i]] = confusion[actual_class[i]][pred_class[i]] + 1
        fig1 = plt.figure()
        max_count = np.amax(confusion)
        print('maxcount %d' %max_count) # Getting sense of vmax to be set
        plt.imshow(confusion, interpolation = 'none', vmin = 0, vmax = 50)
        cbar1 = plt.colorbar(orientation = 'horizontal', fraction = 0.041)
        cbar1.set_label('No. of images')
        plt.ylabel('Actual class label')
        plt.xlabel('Predicted class label')
        plt.xticks(np.arange(0, nb_classes, step = 100))
        plt.yticks(np.arange(0, nb_classes, step = 100))
        plt.title('Unperturbed inputs')

        adv_pred_val = model_eval_layer_adv_imagenet(sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, args=eval_params, nb_samples=nb_samples, attacker=attacker, attack_params=attack_params)
        if not combinedThree and not combinedMultiple:
            adv_pred_class = np.argmax(adv_pred_val, axis=-1)  
        else:
            adv_pred_class = adv_pred_val
        
        adv_confusion = np.zeros([nb_classes, nb_classes])
        for i in range(len(actual_class)):
            adv_confusion[actual_class[i]][adv_pred_class[i]] = adv_confusion[actual_class[i]][adv_pred_class[i]] + 1
        fig2 = plt.figure()
        plt.imshow(adv_confusion, interpolation = 'none', vmin = 0, vmax = 50)
        cbar2 = plt.colorbar(orientation = 'horizontal', fraction = 0.041)
        cbar2.set_label('No. of images')
        plt.ylabel('Actual class label')
        plt.xlabel('Predicted class label')
        plt.xticks(np.arange(0, nb_classes, step = 100))
        plt.yticks(np.arange(0, nb_classes, step = 100))
        plt.title('Adversarial inputs')
        plt.show()
    

    ###########################################################################
    ### Measuring sparsity in the activations
    if act_sparsity and not combinedThree and not combinedMultiple:
        layer_names = model.get_layer_names()
        activation = x

        for l in layer_names:
            inp_activation = activation # Output values of previous layer
            activation = model.get_layer(x, reuse=False, layer=l)
            if 'Conv' in l: # Look at input activation for conv layer
                inp_activation_val, Y_val = model_eval_layer_imagenet_singlebatch(sess, x, y, inp_activation, test_iterator, test_x, test_y, phase=phase, args=eval_params)
                num_examples, num_rows, num_cols, num_filters = inp_activation_val.shape
                print ('layer %s activations num_examples=%d num_rows=%d num_cols=%d num_filters=%d' %(l, num_examples, num_rows, num_cols, num_filters))
                num_sparse = 0
                for i in range(num_examples):
                    for j in range(num_rows):
                        for k in range(num_cols):
                            for kk in range(num_filters):
                                if inp_activation_val[i,j,k,kk] == 0:
                                    num_sparse = num_sparse + 1
                sparse_frac = num_sparse / (num_examples*num_rows*num_cols*num_filters)
                print ("sparse_frac is %f" %sparse_frac)
            if 'Linear' in l: # Look at input activation for f.c. layer
                inp_activation_val, Y_val = model_eval_layer_imagenet_singlebatch(sess, x, y, inp_activation, test_iterator, test_x, test_y, phase=phase, args=eval_params)
                num_examples, num_rows = inp_activation_val.shape
                print ('layer %s num_examples=%d activations num_rows=%d ' %(l, num_examples, num_rows))
                num_sparse = 0
                for i in range(num_examples):
                    for j in range(num_rows):
                        if inp_activation_val[i,j] == 0:
                            num_sparse = num_sparse + 1
                sparse_frac = num_sparse / (num_examples*num_rows)
                print ("sparse_frac is %f" %sparse_frac)

    ## Measuring sparsity in the weights
    if wt_sparsity:
        trained_vars = tf.trainable_variables()
        weight_vars = [v for v in tf.trainable_variables() if "init/k:" in v.name][0]
        print(trained_vars)        
        print(weight_vars)        
    ###########################################################################

    # Close TF session
    sess.close()


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    # Generic flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to save or load model')
    par.add_argument('--data_dir', help='Path to training data',
                     default='/scratch/gallowaa/cifar10/cifar10_data')

    # Architecture and training specific flags
    par.add_argument('--nb_epochs', type=int, default=6,
                     help='Number of epochs to train model')
    par.add_argument('--nb_filters', type=int, default=32,
                     help='Number of filters in first layer')
    par.add_argument('--batch_size', type=int, default=128,
                     help='Size of training batches')
    par.add_argument('--learning_rate', type=float, default=0.001,
                     help='Learning rate')
    par.add_argument('--binary', help='Use a binary model?',
                     action="store_true")
    par.add_argument('--scale', help='Scale activations of the binary model?',
                     action="store_true")
    #### SANCHARI: copied from mnist_attack.py
    par.add_argument('--rand', help='Stochastic weight layer?',
                     action="store_true")
    #### SANCHARI: extra flags to specify if only activations or weights are binarized
    par.add_argument('--binaryactv', help='Use model with binary activations but full precision weights',
                     action="store_true")
    par.add_argument('--binaryweights', help='Use model with binary weights but full precision activations',
                     action="store_true")
    #### SANCHARI: extra flags to specify other low precision CNNs
    par.add_argument('--lowprecision', help='Use other low precision models', action="store_true")
    par.add_argument('--lowprecision1stLayer', help='Use other low precision models which also quantize the first conv layer', action="store_true")
    par.add_argument('--wbits', type=int, default=0, help='No. of bits in weight representation')
    par.add_argument('--abits', type=int, default=0, help='No. of bits in activation representation')
    par.add_argument('--wbitsList', type=int, nargs='+', help='List of No. of bits in weight representation for different layers')
    par.add_argument('--abitsList', type=int, nargs='+', help='List of No. of bits in activation representation for different layers')
    par.add_argument('--stocRound', help='Stochastic rounding for weights (only in training) and activations?', action="store_true")
    par.add_argument('--useSeparateSeed', help='Using a seed other than the default set_random_seed for initializing the lowprecision model', action="store_true") 
    par.add_argument('--seed', type=int, default=1, help='Setting a seed other than the default set_random_seed for initializing the lowprecision model') 
    par.add_argument('--qLinear', help='Indicates whether the linear layers have to be quantized', action="store_true")
    ####
    #### SANCHARI: extra flag to form a combined CNN with both low and high precisions
    par.add_argument('--combined', help='Use use a combined version of full precision and a low precsion model', action="store_true") # For DoReFa net style quantization
    par.add_argument('--attackFPmodel', help='generate attacks on the FP model', action="store_true")  
    par.add_argument('--model_path1', help='Path where saved model1 is stored and can be loaded')
    par.add_argument('--model_path2', help='Path where saved model2 is stored and can be loaded')
    par.add_argument('--avg', help='Use a average function to combine the outputs of the two models', action="store_true") 
    par.add_argument('--weightedAvg', help='Use a weighted average function to combine the outputs of the two models', action="store_true")
    par.add_argument('--weightAlpha', type=float, default=0, help='alpha weight factor for combining, final prob = alpha*FP + (1-alpha)*LP')
    par.add_argument('--combinedTrainable', help='Use a combined version of full precision and a low precsion model that can be attacked directly and potentially trained', action="store_true") # For DoReFa net style quantization
    par.add_argument('--combinedTwoLP', help='Use a combined version of 2 low precision models that can be attacked directly', action="store_true") # For DoReFa net style quantization
    par.add_argument('--combinedThree', help='Use a combined version of full precision and two low precision models that can be attacked directly and potentially trained', action="store_true") # For DoReFa net style quantization
    par.add_argument('--model_path3', help='Path where saved model3 in case of combinedThree model is stored and can be loaded')
    par.add_argument('--wbits2', type=int, default=0, help='No. of bits in weight representation of model2, model1 specified using wbits')
    par.add_argument('--abits2', type=int, default=0, help='No. of bits in activation representation of model2, model2 specified using abits')
    par.add_argument('--wbits2List', type=int, nargs='+', help='List of No. of bits in weight representation for different layers of model2')
    par.add_argument('--abits2List', type=int, nargs='+', help='List of No. of bits in activation representation for different layers of model2')

    par.add_argument('--combinedMultiple', help='Use a combined version of full precision and arbitray number of low precision models', action="store_true") 
    par.add_argument('--numLPmodels', type=int, default=1, help='No. of low precision models part of the combined model')
    par.add_argument('--numFPmodels', type=int, default=1, help='No. of full precision models part of the combined model')
    par.add_argument('--wbitsModels', type=int, nargs='+', help='List of No. of bits in weight representation for the different LP models in the combined model')
    par.add_argument('--abitsModels', type=int, nargs='+', help='List of No. of bits in activation representation for the different LP models in the combined model')
    par.add_argument('--LPmodelPaths', nargs='+', help='Paths for the different LP models in the combined model')
    par.add_argument('--FPmodelPaths', nargs='+', help='Paths for the different FP models in the combined model')
    ####
    #### SANCHARI: extra flag to add mechanism to just test on trained model
    par.add_argument('--onlyTest', help='Simply test on the trained model provided by model_path', action="store_true") 
    #### SANCHARI: extra flag to add retrain from the trained model
    par.add_argument('--retrain', help='retrain on the trained model provided by model_path', action="store_true") 
    par.add_argument('--retrainSavePath', help='path to store the retrained model') 
    par.add_argument('--start_epoch', type=int, default=0, help='starting epoch while retraining') 
    #### SANCHARI: extra flags for input gradient regularization
    par.add_argument('--inpgradreg', help='Train the model using input gradient regularization', action="store_true") 
    par.add_argument('--l2dbl', type=int, default=0, help='l2 double backprop penalty')
    par.add_argument('--l2cs', type=int, default=0, help='l2 certainty sensitivity penalty')
    ####
    #### SANCHARI: extra flags for measuring sparsity 
    par.add_argument('--act_sparsity', help='Measure sparsity of activations', action="store_true")
    par.add_argument('--wt_sparsity', help='Measure sparsity of weights', action="store_true")
    ####

    # Attack specific flags
    par.add_argument('--eps', type=float, default=0.3,
                     help='epsilon')
    par.add_argument('--attack', type=int, default=0,
                     help='Attack type, 0=CW, 2=FGSM')
    par.add_argument('--attack_iterations', type=int, default=100,
                     help='Number of iterations to run CW attack; 1000 is good')
    par.add_argument('--nb_samples', type=int,
                     default=10, help='Nb of inputs to attack')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    # # Adversarial training flags: Default
    # par.add_argument(
    #     '--adv', help='Do MadryEtAl adversarial training?', action="store_true")
    # par.add_argument('--delay', type=int,
    #                  default=0, help='Nb of epochs to delay adv training by')
    # Adversarial training flags copied from mnist_attack.py
    par.add_argument(
        '--adv', help='Adversarial training type?', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=10, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int,
                     default=40, help='Nb of iterations of PGD')

    ### SANCHARI: extra flags to measure other statistics
    par.add_argument('--confusion_data', help='Prints the original and predicted class values needed to generate confusion matrix', action="store_true")

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import cifar10
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
from cleverhans.utils_tf import model_train, model_eval, model_eval_ensemble, batch_eval, tf_model_load
from cleverhans.utils_tf import model_train_teacher, model_train_student, model_train_inpgrad_reg #for training with input gradient regularization

FLAGS = flags.FLAGS

ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_BASICITER = 4
MAX_BATCH_SIZE = 100
MAX_BATCH_SIZE = 100

# enum adversarial training types
ADVERSARIAL_TRAINING_MADRYETAL = 1
ADVERSARIAL_TRAINING_FGSM = 2
MAX_EPS = 0.3 

# Scaling input to softmax
INIT_T = 1.0
#ATTACK_T = 1.0
ATTACK_T = 0.25

def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test
    #return X_train, y_train, X_test, y_test

def data_cifar10_std():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    '''
    X_train /= 255
    X_test /= 255
    '''
    '''
    X_train -= X_train.mean()
    X_train /= X_train.std()
    X_test -= X_test.mean()
    X_test /= X_test.std()
    '''
    e = 1. / np.sqrt(img_rows * img_rows * 3)
    for i in range(X_train.shape[0]):
        X_train[i] = (X_train[i] - np.mean(X_train[i])) / \
            np.maximum(np.std(X_train[i]), e)

    for i in range(X_test.shape[0]):
        X_test[i] = (X_test[i] - np.mean(X_test[i])) / \
            np.maximum(np.std(X_test[i]), e)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test
    #return X_train, y_train, X_test, y_test


def main(argv=None):
    """
    CIFAR10 CleverHans tutorial
    :return:
    """

    # CIFAR10-specific dimensions
    img_rows = 32
    img_cols = 32
    channels = 3
    nb_classes = 10

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

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    phase = tf.placeholder(tf.bool, name="phase")
    logits_scalar = tf.placeholder_with_default(
        INIT_T, shape=(), name="logits_temperature")


    model_path = FLAGS.model_path
    targeted = True if FLAGS.targeted else False
    learning_rate = FLAGS.learning_rate
    nb_filters = FLAGS.nb_filters
    batch_size = FLAGS.batch_size
    nb_samples = FLAGS.nb_samples
    nb_epochs = FLAGS.nb_epochs
    delay = FLAGS.delay
    eps = FLAGS.eps
    adv = FLAGS.adv

    attack = FLAGS.attack
    attack_iterations = FLAGS.attack_iterations
    nb_iter = FLAGS.nb_iter
   
    #### EMPIR extra flags
    lowprecision=FLAGS.lowprecision
    abits=FLAGS.abits
    wbits=FLAGS.wbits
    abitsList=FLAGS.abitsList
    wbitsList=FLAGS.wbitsList
    stocRound=True if FLAGS.stocRound else False
    rand=FLAGS.rand 
    model_path2 = FLAGS.model_path2
    model_path1 = FLAGS.model_path1
    model_path3 = FLAGS.model_path3
    ensembleThree=True if FLAGS.ensembleThree else False
    abits2=FLAGS.abits2
    wbits2=FLAGS.wbits2
    abits2List=FLAGS.abits2List
    wbits2List=FLAGS.wbits2List
    inpgradreg = True if FLAGS.inpgradreg else False
    distill = True if FLAGS.distill else False
    student_epochs = FLAGS.student_epochs
    l2dbl = FLAGS.l2dbl
    l2cs = FLAGS.l2cs
    ####

    save = False
    train_from_scratch = False
    
    if ensembleThree: 
        if (model_path1 is None or model_path2 is None or model_path3 is None):
            train_from_scratch = True
        else:
            train_from_scratch = False
    elif model_path is not None:
        if os.path.exists(model_path):
            # check for existing model in immediate subfolder
            if any(f.endswith('.meta') for f in os.listdir(model_path)):
                train_from_scratch = False
            else:
                model_path = build_model_save_path(
                    model_path, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay)
                print(model_path)
                save = True
                train_from_scratch = True
    else:
        train_from_scratch = True  # train from scratch, but don't save since no path given

    if ensembleThree: 
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified for first model
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the first model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec1 = 1
       elif (len(wbitsList) != 3) or (len(abitsList) != 3):
           print("Error: Need to specify the precisions for activations and weights for the atleast the three convolutional layers of the first model")  
           sys.exit(1)
       else: 
           fixedPrec1 = 0
       
       if (wbits2List is None) or (abits2List is None): # Layer wise separate quantization not specified for second model
           if (wbits2==0) or (abits2==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the second model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec2 = 1
       elif (len(wbits2List) != 3) or (len(abits2List) != 3):
           print("Error: Need to specify the precisions for activations and weights for the atleast the three convolutional layers of the second model")  
           sys.exit(1)
       else: 
           fixedPrec2 = 0

       if (fixedPrec2 != 1) or (fixedPrec1 != 1): # Atleast one of the models have separate precisions per layer
           fixedPrec=0
           print("Within atleast one model has separate precisions")
           if (fixedPrec1 == 1): # first layer has fixed precision
               abitsList = (abits, abits, abits)
               wbitsList = (wbits, wbits, wbits)
           if (fixedPrec2 == 1): # second layer has fixed precision
               abits2List = (abits2, abits2, abits2)
               wbits2List = (wbits2, wbits2, wbits2)
       else:
           fixedPrec=1
       
       if (train_from_scratch):
           print ("The ensemble model cannot be trained from scratch")
           sys.exit(1)
       if fixedPrec == 1:
           from cleverhans_tutorials.tutorial_models import make_ensemble_three_cifar_cnn
           model = make_ensemble_three_cifar_cnn(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbits, abits, wbits2, abits2, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters) 
       else:
           from cleverhans_tutorials.tutorial_models import make_ensemble_three_cifar_cnn_layerwise
           model = make_ensemble_three_cifar_cnn_layerwise(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbitsList, abitsList, wbits2List, abits2List, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters) 
    elif lowprecision:
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers have to specified using wbits and abits flags")
               sys.exit(1)
           else:
               fixedPrec = 1
       elif (len(wbitsList) != 3) or (len(abitsList) != 3):
           print("Error: Need to specify the precisions for activations and weights for the atleast the three convolutional layers")  
           sys.exit(1)
       else: 
           fixedPrec = 0
       
       if fixedPrec:
           from cleverhans_tutorials.tutorial_models import make_basic_lowprecision_cifar_cnn
           model = make_basic_lowprecision_cifar_cnn(
               phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, stocRound=stocRound)  
       else:
           from cleverhans_tutorials.tutorial_models import make_layerwise_lowprecision_cifar_cnn
           model = make_layerwise_lowprecision_cifar_cnn(
               phase, logits_scalar, 'lp_', wbitsList, abitsList, input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, stocRound=stocRound)  
    elif distill:
      from cleverhans_tutorials.tutorial_models import make_distilled_cifar_cnn
      model = make_distilled_cifar_cnn(phase, logits_scalar,
              'teacher_fp_', 'fp_', nb_filters=nb_filters, input_shape=(None, img_rows, img_cols, channels))  
    ####
    else:
        from cleverhans_tutorials.tutorial_models import make_basic_cifar_cnn
        model = make_basic_cifar_cnn(phase, logits_scalar, 'fp_', input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters)


    # separate predictions of teacher for distilled training
    if distill:
        teacher_preds = model.teacher_call(x, reuse=False)
        teacher_logits = model.get_teacher_logits(x, reuse=False)

    # separate calling function for ensemble models
    if ensembleThree:
        preds = model.ensemble_call(x, reuse=False)
    else:
    ##default
        preds = model(x, reuse=False)
    print("Defined TensorFlow model graph.")

    rng = np.random.RandomState([2017, 8, 30])

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        if ensembleThree:
            acc = model_eval_ensemble(
                sess, x, y, preds, X_test, Y_test, phase=phase, args=eval_params)
        else:
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, phase=phase, args=eval_params)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an CIFAR10 model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_name': 'train loss',
        'filename': 'model',
        'reuse_global_step': False,
        'train_scope': 'train',
        'is_training': True
    }
    
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

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

        # do clean training for 'nb_epochs' or 'delay' epochs
        if distill:
            temperature = 10 # 1 means the teacher predictions are used as it is
            teacher_scaled_preds_val = model_train_teacher(sess, x, y, teacher_preds, teacher_logits, 
                        temperature, X_train, Y_train, phase=phase, args=train_params, rng=rng)
            eval_params = {'batch_size': batch_size}
            teacher_acc = model_eval(
                sess, x, y, teacher_preds, X_test, Y_test, phase=phase, args=eval_params)
            print('Test accuracy of the teacher model on legitimate examples: %0.4f' % teacher_acc)
            print('Training the student model...')
            student_train_params = {
                'nb_epochs': student_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'loss_name': 'train loss',
                'filename': 'model',
                'reuse_global_step': False,
                'train_scope': 'train',
                'is_training': True
            }
            if save:
                student_train_params.update({'log_dir': model_path})
            y_teacher = tf.placeholder(tf.float32, shape=(None, nb_classes))
            model_train_student(sess, x, y, preds, temperature, X_train, Y_train, y_teacher=y_teacher, 
                        teacher_preds=teacher_scaled_preds_val, alpha=0.3, beta=0.7, phase=phase, evaluate=evaluate, args=student_train_params, save=save, rng=rng)
        elif inpgradreg: 
            model_train_inpgrad_reg(sess, x, y, preds, X_train, Y_train, phase=phase,
                        evaluate=evaluate, l2dbl = l2dbl, l2cs = l2cs, args=train_params, save=save, rng=rng)
        else: 
            # do clean training for 'nb_epochs' or 'delay' epochs
            model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                        evaluate=evaluate, args=train_params, save=save, rng=rng)

        # optionally do additional adversarial training
        if adv:
            print("Adversarial training for %d epochs" % (nb_epochs - delay))
            train_params.update({'nb_epochs': nb_epochs - delay})
            train_params.update({'reuse_global_step': True})
            model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                        predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params,
                        save=save, rng=rng)

    else:
        if ensembleThree: 
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            stored_variables = ['lp_conv1_init/k', 'lp_conv2_init/k', 'lp_conv3_init/k', 'lp_ip1init/W', 'lp_logits_init/W']
            variable_dict = dict(zip(stored_variables, variables[:5])) 
            # Restore the first set of variables from model_path1
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Restore the second set of variables from model_path2
            variable_dict = dict(zip(stored_variables, variables[5:10]))
            saver2 = tf.train.Saver(variable_dict)
            saver2.restore(sess, tf.train.latest_checkpoint(model_path2))
            stored_variables = ['fp_conv1_init/k', 'fp_conv2_init/k', 'fp_conv3_init/k', 'fp_ip1init/W', 'fp_logits_init/W']
            variable_dict = dict(zip(stored_variables, variables[10:]))
            saver3 = tf.train.Saver(variable_dict)
            saver3.restore(sess, tf.train.latest_checkpoint(model_path3))
        else:
            tf_model_load(sess, model_path)
            print('Restored model from %s' % model_path)
        evaluate()

    # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    if ensembleThree: 
        accuracy = model_eval_ensemble(sess, x, y, preds, X_test, Y_test, phase=phase, feed={phase: False}, args=eval_params)
    else:
        accuracy = model_eval(sess, x, y, preds, X_test, Y_test, phase=phase,
                          feed={phase: False}, args=eval_params)

    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Build dataset
    ###########################################################################

    if targeted:
        from cleverhans.utils import build_targeted_dataset
        adv_inputs, true_labels, adv_ys = build_targeted_dataset(
            X_test, Y_test, np.arange(nb_samples), nb_classes, img_rows, img_cols, channels)
    else:
        adv_inputs = X_test[:nb_samples]

    ###########################################################################
    # Craft adversarial examples using generic approach
    ###########################################################################
    if targeted:
        att_batch_size = np.clip(
            nb_samples * (nb_classes - 1), a_max=MAX_BATCH_SIZE, a_min=1)
        nb_adv_per_sample = nb_classes - 1
        yname = "y_target"

    else:
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
        nb_adv_per_sample = 1
        adv_ys = None
        yname = "y"

    print('Crafting ' + str(nb_samples) + ' * ' + str(nb_adv_per_sample) +
          ' adversarial examples')
    print("This could take some time ...")

    if ensembleThree:
        model_type = 'ensembleThree'
    else:
        model_type = 'default'

    if attack == ATTACK_CARLINI_WAGNER_L2:
        from cleverhans.attacks import CarliniWagnerL2
        attacker = CarliniWagnerL2(model, back='tf', model_type=model_type, num_classes=nb_classes, sess=sess)
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.1,
                         'batch_size': att_batch_size,
                         'initial_const': 10,
                         }
    elif attack == ATTACK_JSMA:
        from cleverhans.attacks import SaliencyMapMethod
        attacker = SaliencyMapMethod(model, back='tf', model_type=model_type, sess=sess, num_classes=nb_classes)
        attack_params = {'theta': 1., 'gamma': 0.1}
    elif attack == ATTACK_FGSM:
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, back='tf', model_type=model_type, sess=sess, num_classes=nb_classes)
        attack_params = {'eps': eps}
    elif attack == ATTACK_MADRYETAL:
        from cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, back='tf', model_type=model_type, sess=sess, num_classes=nb_classes)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    elif attack == ATTACK_BASICITER:
        from cleverhans.attacks import BasicIterativeMethod
        attacker = BasicIterativeMethod(model, back='tf', sess=sess, model_type=model_type, num_classes=nb_classes)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    else:
        print("Attack undefined")
        sys.exit(1)

    attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
    X_test_adv = attacker.generate_np(adv_inputs, phase, **attack_params)
    '''
    adv_x = attacker.generate(x, phase, **attack_params)
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    eval_params = {'batch_size': att_batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [adv_inputs], feed={
                             phase: False}, args=eval_params)
    '''

    if targeted:
        assert X_test_adv.shape[0] == nb_samples * \
            (nb_classes - 1), X_test_adv.shape
        # Evaluate the accuracy of the CIFAR10 model on adversarial examples
        print("Evaluating targeted results")
        adv_accuracy = model_eval(sess, x, y, preds, X_test_adv, true_labels,
                                  phase=phase, args=eval_params)
    else:
        # assert X_test_adv.shape[0] == nb_samples, X_test_adv.shape
        # Evaluate the accuracy of the CIFAR10 model on adversarial examples
        print("Evaluating un-targeted results")
        if ensembleThree:
            adv_accuracy = model_eval_ensemble(sess, x, y, preds, X_test_adv, Y_test,
                                  phase=phase, args=eval_params)
        else: #default below
            adv_accuracy = model_eval(sess, x, y, preds, X_test_adv, Y_test,
                                      phase=phase, args=eval_params)

    # Compute the number of adversarial examples that were successfully found
    print('Test accuracy on adversarial examples {0:.4f}'.format(adv_accuracy))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((X_test_adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Friendly output for pasting into spreadsheet
    print('{0:.4f},'.format(accuracy))
    print('{0:.4f},'.format(adv_accuracy))
    print('{0:.4f},'.format(percent_perturbed))

    sess.close()

    '''
    print("Repeating the process, using adversarial training")

    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained CIFAR10 model on
        # legitimate test examples
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                              phase=phase,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained CIFAR10 model on
        # adversarial examples
        accuracy_adv = model_eval(sess, x, y, preds_adv, X_test,
                                  Y_test, phase=phase, args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Perform adversarial training
    train_params.update({'reuse_global_step': True})
    model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                predictions_adv=preds_adv, evaluate=evaluate_2,
                args=train_params)
    '''
    '''
    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test,
                          phase=phase,
                          args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))
    '''


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
    par.add_argument('--rand', help='Stochastic weight layer?',
                     action="store_true")

    # Attack specific flags
    par.add_argument('--eps', type=float, default=0.1,
                     help='epsilon')
    par.add_argument('--attack', type=int, default=0,
                     help='Attack type, 0=CW, 2=FGSM')
    par.add_argument('--attack_iterations', type=int, default=100,
                     help='Number of iterations to run CW attack; 1000 is good')
    par.add_argument('--nb_samples', type=int,
                     default=10000, help='Nb of inputs to attack')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    # Adversarial training flags
    par.add_argument(
        '--adv', help='Adversarial training type?', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=10, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int,
                     default=40, help='Nb of iterations of PGD')
    
    # EMPIR specific flags
    par.add_argument('--lowprecision', help='Use other low precision models', action="store_true")
    par.add_argument('--wbits', type=int, default=0, help='No. of bits in weight representation')
    par.add_argument('--abits', type=int, default=0, help='No. of bits in activation representation')
    par.add_argument('--wbitsList', type=int, nargs='+', help='List of No. of bits in weight representation for different layers')
    par.add_argument('--abitsList', type=int, nargs='+', help='List of No. of bits in activation representation for different layers')
    par.add_argument('--stocRound', help='Stochastic rounding for weights (only in training) and activations?', action="store_true")
    par.add_argument('--model_path1', help='Path where saved model1 is stored and can be loaded')
    par.add_argument('--model_path2', help='Path where saved model2 is stored and can be loaded')
    par.add_argument('--ensembleThree', help='Use an ensemble of full precision and two low precision models that can be attacked directly', action="store_true") 
    par.add_argument('--model_path3', help='Path where saved model3 in case of combinedThree model is stored and can be loaded')
    par.add_argument('--wbits2', type=int, default=0, help='No. of bits in weight representation of model2, model1 specified using wbits')
    par.add_argument('--abits2', type=int, default=0, help='No. of bits in activation representation of model2, model2 specified using abits')
    par.add_argument('--wbits2List', type=int, nargs='+', help='List of No. of bits in weight representation for different layers of model2')
    par.add_argument('--abits2List', type=int, nargs='+', help='List of No. of bits in activation representation for different layers of model2')
    # extra flags for defensive distillation
    par.add_argument('--distill', help='Train the model using distillation', action="store_true") 
    par.add_argument('--student_epochs', type=int, default=50, help='No. of epochs for which the student model is trained')
    # extra flags for input gradient regularization
    par.add_argument('--inpgradreg', help='Train the model using input gradient regularization', action="store_true") 
    par.add_argument('--l2dbl', type=int, default=0, help='l2 double backprop penalty')
    par.add_argument('--l2cs', type=int, default=0, help='l2 certainty sensitivity penalty')
    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()

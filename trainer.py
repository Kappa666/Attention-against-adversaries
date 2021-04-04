#model trainer

import os
import time
import glimpse
import warnings
import argparse
import datasets
from datasets import FILE_PATH
import model_backbone
import attack_backbone

import numpy as np
import tensorflow as tf

from functools import partial
from datasets import _preprocess_y
from math import ceil

#input args
parser = argparse.ArgumentParser()

parser.add_argument('--name')
parser.add_argument('--model')
parser.add_argument('--dataset', default='imagenet10')

parser.add_argument('--augment')
parser.add_argument('--sampling')
parser.add_argument('--coarse_fixations')
parser.add_argument('--auxiliary')

parser.add_argument('--single_scale', default=0)
parser.add_argument('--scale4_freeze', default=0)
parser.add_argument('--branched_network', default=0)
parser.add_argument('--upsample_fixations', default=0)
parser.add_argument('--blur_fixations', default=0)
parser.add_argument('--pooling', default='None')
parser.add_argument('--dropout', default=0)
parser.add_argument('--cifar_ecnn', default=0)
parser.add_argument('--cifar_feedback', default=0)
parser.add_argument('--mnist_attention', default=0)
parser.add_argument('--mnist_dummy_attention', default=0)
parser.add_argument('--mnist_dummy_scaled_attention', default=0)
parser.add_argument('--mnist_restricted_attention', default=0)
parser.add_argument('--mnist_retinal_attention', default=0)
parser.add_argument('--adv_train', default=0)
parser.add_argument('--epochs', default=100)
parser.add_argument('--shared', default=0)
parser.add_argument('--num_transformers', default=5)

parser.add_argument('--only_evaluate', default=0)
args = vars(parser.parse_args())

name = str(args['name'])
model = str(args['model'])
dataset = str(args['dataset'])
sampling  = bool(int(args['sampling']))
coarse_fixations = bool(int(args['coarse_fixations']))
auxiliary = bool(int(args['auxiliary']))
augment = bool(int(args['augment']))
single_scale = bool(int(args['single_scale']))
scale4_freeze = bool(int(args['scale4_freeze']))
branched_network = bool(int(args['branched_network']))
upsample_fixations = bool(int(args['upsample_fixations']))
blur_fixations = bool(int(args['blur_fixations']))
pooling = str(args['pooling'])
dropout = bool(int(args['dropout']))
only_evaluate = bool(int(args['only_evaluate']))
cifar_ecnn = bool(int(args['cifar_ecnn']))
cifar_feedback = bool(int(args['cifar_feedback']))
mnist_attention = bool(int(args['mnist_attention']))
mnist_dummy_attention = bool(int(args['mnist_dummy_attention']))
mnist_dummy_scaled_attention = bool(int(args['mnist_dummy_scaled_attention']))
mnist_restricted_attention = bool(int(args['mnist_restricted_attention']))
mnist_retinal_attention = bool(int(args['mnist_retinal_attention']))
adv_train = bool(int(args['adv_train']))
epochs = int(args['epochs'])
shared = bool(int(args['shared']))
num_transformers = int(args['num_transformers'])

pooling = None if pooling == 'None' else pooling 
scales = 'scale4' if single_scale else 'all'

if dataset == 'test10':
        warnings.warn('running in test mode!')

save_file = '{}/model_checkpoints/{}.h5'.format(FILE_PATH, name)

if only_evaluate:
        if not os.path.exists(save_file):
                raise ValueError
else:
        if os.path.exists(save_file):
                raise ValueError

print(save_file)

distribution = tf.distribute.MirroredStrategy()

if dataset == 'cifar10' or dataset == 'imagenet10' or dataset == 'bbox_imagenet10' or dataset == 'cluttered_mnist':
        num_classes = 10
elif dataset == 'imagenet100':
        num_classes = 100
elif dataset == 'imagenet':
        num_classes = 1000
else:
        raise ValueError

with distribution.scope():      
        
        print('Building network...')

        model_tag = model
        #build network
        if model == 'resnet':
                #check params
                assert(auxiliary is False)
                assert(single_scale is False)
                assert(scale4_freeze is False)
                assert(not (upsample_fixations and blur_fixations))

                if coarse_fixations:
                        if upsample_fixations:
                                base_model_input_shape = (320,320,3)
                        elif blur_fixations:
                                base_model_input_shape = (240,240,3)
                        else:
                                base_model_input_shape = (224,224,3)
                else:
                        base_model_input_shape = (320,320,3)

                model = model_backbone.resnet(base_model_input_shape=base_model_input_shape, num_classes=num_classes, augment=augment, sampling=sampling, coarse_fixations=coarse_fixations, coarse_fixations_upsample=upsample_fixations, coarse_fixations_gaussianblur=blur_fixations, branched_network=branched_network)
                if only_evaluate:
                        model.load_weights(save_file, by_name=True)

        elif model == 'resnet_cifar':
                #check params
                assert(auxiliary is False)
                assert(scale4_freeze is False)

                if single_scale:
                        assert(cifar_ecnn)
        
                if adv_train:
                        return_logits=True
                else:
                        return_logits=False

                if coarse_fixations:
                        if upsample_fixations:
                                base_model_input_shape = (32, 32, 3)
                        else:
                                base_model_input_shape = (24, 24, 3)
                else:
                        if cifar_ecnn:
                                base_model_input_shape = (12, 12, 3)
                        else:
                                base_model_input_shape = (32, 32, 3)

                if adv_train and (not only_evaluate):
                        adv_training=True
                        assert(not dropout)
                else:
                        adv_training=False
            
                model = model_backbone.resnet_cifar(base_model_input_shape=base_model_input_shape, augment=augment, sampling=sampling, coarse_fixations=coarse_fixations, coarse_fixations_upsample=upsample_fixations, coarse_fixations_gaussianblur=blur_fixations, approx_ecnn=cifar_ecnn, return_logits=return_logits, build_feedback=cifar_feedback, ecnn_pooling=pooling, ecnn_dropout=dropout, ecnn_single_scale=single_scale, adv_training=adv_training, wider_network=adv_train)
                if only_evaluate:
                        #convention for resnet_cifar is to not use the name
                        model.load_weights(save_file, by_name=False)

        elif model == 'attention_mnist':
                #check params
                assert(auxiliary is False)
                assert(single_scale is False)
                assert(scale4_freeze is False)
                assert(blur_fixations is False)
                assert(coarse_fixations is False)
                assert(upsample_fixations is False)
                assert(cifar_ecnn is False)
                assert(cifar_feedback is False)
                assert(sampling is False)

                if adv_train:
                        return_logits=True
                else:
                        return_logits=False             

                model = model_backbone.attention_mnist(augment=augment, return_logits=return_logits, attention=mnist_attention, dummy_attention=mnist_dummy_attention, dummy_scaled_attention=mnist_dummy_scaled_attention, restricted_attention=mnist_restricted_attention, retinal_attention=mnist_retinal_attention)
                if only_evaluate:
                        #convention for mnist models is to not use the name
                        model.load_weights(save_file, by_name=False)

        elif model == 'ecnn':
                #check params
                assert(coarse_fixations is False)
                assert(upsample_fixations is False)
                assert(blur_fixations is False)
                if single_scale:
                        assert(auxiliary is False)

                model = model_backbone.ecnn(num_classes=num_classes, augment=augment, auxiliary=auxiliary, sampling=sampling, scales=scales, pooling=pooling, dropout=dropout, scale4_freeze=scale4_freeze)
                if only_evaluate:
                        model.load_weights(save_file, by_name=True)

        elif model == 'parallel_transformers':
                model = model_backbone.parallel_transformers(num_classes=num_classes, augment=augment, restricted_attention=mnist_restricted_attention, shared=shared, num_transformers=num_transformers)
                if only_evaluate:
                        model.load_weights(save_file, by_name=False)                    
        else:
                raise ValueError

        model.summary()
        t1 = time.time()
        print('Loading dataset...')
        if num_transformers == 5:
                scale = 1.5
        else:
                scale = 1
        #load dataset, set defaults
        if dataset == 'imagenet10' or dataset == 'bbox_imagenet10':
                train_dataset_size = 5659
                test_dataset_size = 500
                #epochs=400
                base_lr=1e-3
                batch_size=128
                checkpoint_interval=999
                optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
                steps_per_epoch = ceil(train_dataset_size / batch_size)
                validation_steps = ceil(test_dataset_size / batch_size)

                if dataset == 'imagenet10':
                        train_dataset, test_dataset = datasets.load_imagenet(data_dir=dataset, only_test=only_evaluate, aux_labels=auxiliary, batch_size=batch_size)
                elif dataset == 'bbox_imagenet10':
                        train_dataset, test_dataset = datasets.load_imagenet10(only_test=only_evaluate, only_bbox=True, aux_labels=auxiliary, batch_size=batch_size)

                def lr_schedule(epoch, lr, base_lr):
                        #keeps learning rate to a schedule
                        if epoch > 360:
                                lr = base_lr * 0.5e-3
                        elif epoch > 320:
                                lr = base_lr * 1e-3
                        elif epoch > 240:
                                lr = base_lr * 1e-2
                        elif epoch > 160:
                                lr = base_lr * 1e-1
                        
                        return lr / scale
        elif dataset == 'imagenet100' or dataset == 'imagenet':
                train_dataset_size = 1281167
                test_dataset_size = 50000
                base_lr=1e-1
                batch_size=256
                checkpoint_interval=999
                optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, decay=1e-4, momentum=0.9)

                if dataset == 'imagenet100':
                        steps_per_epoch = ceil(train_dataset_size / 10 / batch_size)
                        validation_steps = ceil(test_dataset_size / 10 / batch_size)
                else:
                        steps_per_epoch = ceil(train_dataset_size / batch_size)
                        validation_steps = ceil(test_dataset_size / batch_size)

                train_dataset, test_dataset = datasets.load_imagenet(data_dir=dataset, only_test=only_evaluate, aux_labels=auxiliary, batch_size=batch_size)

                if dataset == 'imagenet100':        
                        #epochs = 130
                        def lr_schedule(epoch, lr, base_lr):
                                #keeps learning rate to a schedule

                                if epoch > 120:
                                        lr = base_lr * 0.5e-3
                                elif epoch > 90:
                                        lr = base_lr * 1e-3
                                elif epoch > 60:
                                        lr = base_lr * 1e-2
                                elif epoch > 30:
                                        lr = base_lr * 1e-1

                                return lr / scale
                else:        
                        #epochs = 90
                        def lr_schedule(epoch, lr, base_lr):
                                #keeps learning rate to a schedule
                                if epoch > 80:
                                        lr = base_lr * 1e-3
                                elif epoch > 60:
                                        lr = base_lr * 1e-2
                                elif epoch > 30:
                                        lr = base_lr * 1e-1

                                return lr / scale
        elif dataset == 'cifar10' or dataset == 'integer_cifar10':
                checkpoint_interval=999

                if not adv_train:
                        #epochs=200
                        base_lr=1e-3
                        batch_size=128
                        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
                else:
                        #epochs=165
                        base_lr=0.1
                        batch_size=64
                        optimizer=tf.keras.optimizers.SGD(learning_rate=base_lr, decay=1e-4, momentum=0.9)

                if dataset == 'cifar10':
                        x_train, y_train, x_test, y_test = datasets.load_cifar10(only_test=only_evaluate)
                elif dataset == 'integer_cifar10':
                        x_train, y_train, x_test, y_test = datasets.load_integer_cifar10(only_test=only_evaluate)

                if not adv_train:
                        def lr_schedule(epoch, lr, base_lr):
                                #keeps learning rate to a schedule
                                if epoch > 180:
                                        lr = base_lr * 0.5e-3
                                elif epoch > 160:
                                        lr = base_lr * 1e-3
                                elif epoch > 120:
                                        lr = base_lr * 1e-2
                                elif epoch > 80:
                                        lr = base_lr * 1e-1
                                        
                                return lr / scale
                else:
                        def lr_schedule(epoch, lr, base_lr):
                                #keeps learning rate to a schedule

                                if epoch > 125:
                                        lr = base_lr * 1e-2
                                elif epoch > 85:
                                        lr = base_lr * 1e-1

                                return lr
        elif dataset == 'cluttered_mnist':
                #epochs=100
                base_lr=1e-3
                batch_size=128
                checkpoint_interval=999
                optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

                x_train, y_train, x_test, y_test = datasets.load_cluttered_mnist(only_test=only_evaluate)

                def lr_schedule(epoch, lr, base_lr):
                        #keeps learning rate to a schedule

                        if epoch > 90:
                                lr = base_lr * 0.5e-3
                        elif epoch > 80:
                                lr = base_lr * 1e-3
                        elif epoch > 60:
                                lr = base_lr * 1e-2
                        elif epoch > 40:
                                lr = base_lr * 1e-1
                                
                        return lr / scale
        elif dataset == 'test10':
                #epochs=3
                base_lr=1e-6
                batch_size=2
                checkpoint_interval=2
                optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

                input_size = 32 if model_tag == 'resnet_cifar' else 320
                x_train, y_train, x_test, y_test = datasets.load_test10(batch_size, input_size=input_size)

                def lr_schedule(epoch, lr, base_lr):
                        return lr
        else:
                raise ValueError
        t2 = time.time()
        print('Time to load:', t2 - t1)
        print('Training model...')
        t1 = time.time()

        if adv_train:
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
                loss = 'categorical_crossentropy'

        if only_evaluate:                               
                model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.), metrics=['accuracy'])
        else:
                lr_schedule_filled = partial(lr_schedule, base_lr=base_lr)
                model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

                #create training callbacks
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule_filled, verbose=1)
                #lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
                oldest_model_saver = tf.keras.callbacks.ModelCheckpoint(filepath=save_file, save_best_only=False, save_weights_only=True, verbose=1)
                #interval_model_saver = tf.keras.callbacks.ModelCheckpoint(filepath='{}/model_checkpoints/{}-'.format(FILE_PATH, name)+'{epoch:03d}.h5', period=checkpoint_interval, save_best_only=False, save_weights_only=True, verbose=1)

                callbacks = [lr_scheduler, oldest_model_saver]

                if dataset == 'imagenet100' or dataset =='imagenet' or dataset == 'imagenet10':
                        #stream from tfrecords
                        #note: does not exactly partition train/test epochs

                        if adv_train:
                                raise NotImplementedError

                        model.fit(train_dataset, steps_per_epoch=steps_per_epoch, validation_data=test_dataset, validation_steps=validation_steps, epochs=epochs, callbacks=callbacks, verbose=1)
                else:
                        #fit directly from memory
                        
                        if adv_train and (dataset != 'cifar10' and dataset != 'cluttered_mnist'):
                                raise NotImplementedError

                        if not adv_train:
                                if not auxiliary:
                                        model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, validation_data=(x_test, y_test), epochs=epochs, callbacks=callbacks, verbose=1)
                                else:
                                        model.fit(x_train, [y_train, y_train, y_train, y_train, y_train], batch_size=batch_size, shuffle=True, validation_data=(x_test, [y_test, y_test, y_test, y_test, y_test]), epochs=epochs, callbacks=callbacks, verbose=1)
                        else:
                                #adversarial training
                                assert(augment is False) #hacky solution, but it ALWAYS augments during training phase
                                if model_tag != 'resnet_cifar' and model_tag != 'attention_mnist':
                                        raise NotImplementedError    

                                @tf.function
                                def train_model_on_batch(x, y):
                                        return distribution.experimental_run_v2(model.train_on_batch, args=(x, y), kwargs={'reset_metrics': False})                                                             

                                #allowed sets for mechanism gaze
                                if model_tag == 'resnet_cifar':
                                        if coarse_fixations:    
                                                if upsample_fixations:    
                                                        gaze_val = 8
                                                else:
                                                        gaze_val = 4
                                        elif sampling:
                                                gaze_val = 8                   
                                        elif cifar_ecnn:
                                                gaze_val = 4   
                                        else:
                                                gaze_val = 0             
                
                                #for each epoch
                                for epoch_i in range(epochs):
                                        #shuffle
                                        shuffled_order = np.arange(len(x_train))
                                        np.random.shuffle(shuffled_order)

                                        assert(len(np.unique(shuffled_order)) == len(shuffled_order))

                                        x_train = x_train[shuffled_order]
                                        y_train = y_train[shuffled_order]

                                        #iterate batch by batch
                                        num_batches = len(x_train) // batch_size
                                        x_batches = np.array_split(x_train, num_batches)
                                        y_batches = np.array_split(y_train, num_batches)


                                        for batch_i, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                                                batch_start = time.time()

                                                #augment the data
                                                x_batch_augmented = glimpse.image_augmentation(x_batch, dataset).numpy()

                                                #choose a point of fixation
                                                if model_tag == 'resnet_cifar':                        
                                                        if gaze_val == 0:
                                                                #dummy val for no gaze
                                                                gaze_x = 999
                                                                gaze_y = 999
                                                        else:
                                                                gaze_x = tf.random.uniform(shape=[], minval=-gaze_val, maxval=gaze_val, dtype=tf.int32).numpy()
                                                                gaze_y = tf.random.uniform(shape=[], minval=-gaze_val, maxval=gaze_val, dtype=tf.int32).numpy()
                                                        gaze = [gaze_x, gaze_y]
                                                else:
                                                        gaze = None
                            
                                                #get adv examples
                                                x_batch_adv = distribution.experimental_run_v2(attack_backbone.perturb, args=(x_batch_augmented,y_batch), kwargs={'model': model, 'gaze': gaze})

                                                #train on adv examples
                                                if gaze is None:                        
                                                        metrics = train_model_on_batch(x_batch_adv, y_batch)
                                                else:
                                                        metrics = train_model_on_batch([ x_batch_adv, np.tile([gaze], (len(x_batch_adv),1)) ], y_batch)

                                                batch_end = time.time()
                                                print('epoch {} batch {}/{} took {:3f}s. with gaze {}: loss {}, acc {}'.format(epoch_i, batch_i, num_batches, batch_end - batch_start, gaze, metrics[0], metrics[1]))

                                        if gaze is None:                                  
                                                metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
                                        else:
                                                print('validation gaze: {}'.format(gaze))
                                                metrics = model.evaluate([ x_test, np.tile([gaze], (len(x_test),1)) ], y_test, batch_size=batch_size, verbose=0)
                                        print('validation loss: {}'.format(metrics[0]))
                                        print('validation acc: {}'.format(metrics[1]))

                                        #save weights every epoch
                                        print('saving model weights ...')
                                        model.save_weights(save_file)

                                        #next step in learning schedule
                                        print('updating learning rate ...')

                                        lr_before = tf.keras.backend.get_value(model.optimizer.lr)
                                        tf.keras.backend.set_value(model.optimizer.lr, lr_schedule(epoch_i, base_lr, base_lr))
                                        lr_after =  tf.keras.backend.get_value(model.optimizer.lr)

                                        print('updated learning rate from {} to {}'.format(lr_before, lr_after))


        #evaluate model
        #repeats by default (sanity check for model stochasticity)
        repeats = 3
        t2 = time.time()
        print("Time to train:", t2 - t1)

        for _ in range(repeats):

                if dataset == 'imagenet100' or dataset == 'imagenet' or dataset=='imagenet10':
                        scores = model.evaluate(test_dataset, steps=validation_steps, verbose=0)
                else:
                        if (not only_evaluate) and adv_train and model_tag == 'resnet_cifar':
                                assert(not auxiliary)                
                                scores = model.evaluate([ x_test, np.tile([[0,0]], (len(x_test),1)) ], y_test, verbose=0)
                        else:
                                if not auxiliary:       
                                        scores = model.evaluate(x_test, y_test, verbose=0)
                                else:
                                        scores = model.evaluate(x_test, [y_test, y_test, y_test, y_test, y_test], verbose=0)

                print('({})Test loss: {}.'.format(_, scores[0]))
                print('({})Test accuracy: {}.'.format(_, scores[1]))

#dataset tools

import os
import re
import pickle
import glob
import attack_backbone

import scipy as sp
import numpy as np
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from functools import partial

FILE_PATH = '/om5/user/kappa666/Attention-against-adversaries'

def load_imagenet(data_dir='imagenet100', only_test=False, aux_labels=False, batch_size=256, subsample=False):
	# imagenet datasets (100 randomly pre-selected classes or full)

	if data_dir != 'imagenet100' and data_dir != 'imagenet' and data_dir != 'imagenet10':
		raise ValueError
	
	if subsample:
		if not only_test:
			raise NotImplementedError
		if data_dir != 'imagenet':
			raise NotImplementedError
		data_dir = data_dir + '_subsampled'    

	shortlist = None
	if data_dir == 'imagenet100':
		shortlist_loc = '{}/imagenet100/shortlist.pickle'.format(FILE_PATH)
		shortlist = pickle.load(open(shortlist_loc, 'rb'))
	if data_dir == 'imagenet10':
		shortlist_loc = '{}/imagenet10/shortlist.pickle'.format(FILE_PATH)
		shortlist = pickle.load(open(shortlist_loc, 'rb'))

	train_cache_pattern = '{}/{}/train-*'.format(FILE_PATH, data_dir)
	test_cache_pattern = '{}/{}/test-*'.format(FILE_PATH, data_dir)

	#if cache does not exist, build
	if not (glob.glob(train_cache_pattern) or glob.glob(test_cache_pattern)):
		#load raw images (resized)
		train_data_gen, test_data_gen = _build_imagenet(data_dir, shortlist)

		_write_records(train_data_gen, 1024, data_dir, 'train')
		_write_records(test_data_gen, 1024, data_dir, 'test')

	#load cache as tf dataset
	train_tfrecords = tf.data.TFRecordDataset.list_files(train_cache_pattern, shuffle=True)
	test_tfrecords = tf.data.TFRecordDataset.list_files(test_cache_pattern, shuffle=False)

	train_dataset = tf.data.TFRecordDataset(train_tfrecords, num_parallel_reads=4)
	test_dataset = tf.data.TFRecordDataset(test_tfrecords, num_parallel_reads=4)

	_parse_function_filled = partial(_parse_function, aux_labels=aux_labels)
	train_dataset = train_dataset.map(map_func=_parse_function_filled, num_parallel_calls=4)
	test_dataset = test_dataset.map(map_func=_parse_function_filled, num_parallel_calls=4)

	train_dataset = train_dataset.map(map_func=_rebuild_image, num_parallel_calls=4)
	test_dataset = test_dataset.map(map_func=_rebuild_image, num_parallel_calls=4)

	train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
	train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False)
	train_dataset = train_dataset.prefetch(3)

	test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)

	if not only_test:
		train_dataset = train_dataset.repeat()
		test_dataset = test_dataset.repeat()

	if only_test:
		return None, test_dataset
	else:
		return train_dataset, test_dataset

def _build_imagenet(data_dir, shortlist):
	# build imagenet tfrecords

	train_data_dir = FILE_PATH + "/imagenet10/train"
	test_data_dir =  FILE_PATH + "/imagenet10/val"

	class_id_to_name = {}
	class_name_to_id = {}	

	if data_dir == 'imagenet100' or data_dir == 'imagenet10':
		train_data_dir_files = [i for i in os.listdir(train_data_dir) if i in shortlist]
		test_data_dir_files = [i for i in os.listdir(test_data_dir) if i in shortlist]
	elif data_dir == 'imagenet':
		train_data_dir_files = [i for i in os.listdir(train_data_dir)]
		test_data_dir_files = [i for i in os.listdir(test_data_dir)]
	else:
		raise ValueError

	for class_id, class_name in enumerate(train_data_dir_files):
		class_id_to_name[class_id] = class_name
		class_name_to_id[class_name] = class_id

	num_classes = len(class_id_to_name)

	#check number of classes
	if data_dir == 'imagenet100':
		assert(num_classes == 100)
	elif data_dir == 'imagenet':
		assert(num_classes == 1000)
	elif data_dir == 'imagenet10':
		assert(num_classes == 10)
	else:
		raise ValueError

	#check that the class names match between train, test and that they are in the map now
	assert(np.all([i in class_id_to_name.values() for i in np.unique(train_data_dir_files)]))
	assert(np.all([i in class_id_to_name.values() for i in np.unique(test_data_dir_files)]))

	test_data_gen = _load_images_gen(test_data_dir, class_name_to_id, data_dir, shortlist, 320) #yields x_test, y_test
	train_data_gen = _load_images_gen(train_data_dir, class_name_to_id, data_dir, shortlist, 320) #yields x_train, y_train
	
	return train_data_gen, test_data_gen

def _load_images_gen(data_dir, class_name_to_id, data_dir_tag, shortlist, size):
	# generator that reads and preprocess image from disk for dumping to tfrecords

	if data_dir_tag == 'imagenet':
		assert(shortlist is None)
	elif data_dir_tag == 'imagenet100' or data_dir_tag == 'imagenet10':
		assert(shortlist is not None)
	else:
		raise ValueError
	
	class_and_image_names = []

	for class_name in os.listdir(data_dir):
		if data_dir_tag == 'imagenet100' or data_dir_tag == 'imagenet10':
			if class_name not in shortlist:
				continue
		for image in os.listdir(os.path.join(data_dir, class_name)):
			class_and_image_names.append((class_name, image))

	np.random.shuffle(class_and_image_names)

	for class_name, image in tqdm(class_and_image_names):
		file_dir = os.path.join(data_dir, class_name, image)

		#the image raw data
		image_data = tf.keras.preprocessing.image.load_img(file_dir, target_size=None, color_mode='rgb')

		#the image label per assignments generated above
		image_label = class_name_to_id[class_name]

		#resize to 320, 320
		image_data = _resize_image(image_data, size)		     

		yield image_data, image_label

def _bytes_feature(value):
	# parse bytes data from tfrecords
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature_jpeg(value):
	# parse jpeg data from tfrecords
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _parse_function(proto, aux_labels):
	# parse data from tfrecords
	keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
			    'label': tf.io.FixedLenFeature([], tf.string)}
	#'label_aux': tf.io.FixedLenFeature([], tf.string)
	
	parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
	parsed_features['image'] = tf.io.decode_jpeg(parsed_features['image'])	  
	parsed_features['label'] = tf.io.decode_raw(parsed_features['label'], tf.float32)
	#parsed_features['label_aux'] = tf.io.decode_raw(parsed_features['label_aux'], tf.float32)
    
	if not aux_labels:
		return parsed_features['image'], parsed_features['label']
	else:	 
		return parsed_features['image'], (parsed_features['label'], parsed_features['label'], parsed_features['label'], parsed_features['label'], parsed_features['label'])

def _rebuild_image(image, label):
	# rebuild image read from a tfrecords
	image = tf.reshape(image, [320, 320, 3])
	image = tf.cast(image, dtype=tf.float32)
	image = image/255.
	return image, label

def _write_records(data_gen, files_per_shard, data_dir, name):
	# write tfrecords

	if data_dir == 'imagenet100':
		num_classes=100
	elif data_dir == 'imagenet':
		num_classes=1000
	elif data_dir == 'imagenet10':
		num_classes=10
	else:
		raise ValueError

	container_x = []
	container_y = []
	x_count = 0
	y_count = 0
	current_shard_id = 0

	for x, y in data_gen:

		container_x.append(x)
		container_y.append(y)

		if len(container_x) == files_per_shard:

			_write_records_helper(container_x, container_y, data_dir, name, current_shard_id, num_classes)

			container_x = []
			container_y = []
			x_count += files_per_shard
			y_count += files_per_shard
			current_shard_id += 1

	#leftovers
	if len(container_x) != 0:
		_write_records_helper(container_x, container_y, data_dir, name, current_shard_id, num_classes)

		assert(len(container_x) == len(container_y))
		x_count += len(container_x)
		y_count += len(container_y)

	print('final x sample count: {}'.format(x_count))
	print('final y sample count: {}'.format(y_count))

def _write_records_helper(container_x, container_y, data_dir, name, current_shard_id, num_classes):
	# write tfrecords internals

	container_x = np.array(container_x)
	container_y = np.array(container_y)	

	#preprocessing for x is handled instead by tf dataset to get around jpeg conversions
	#container_x = _preprocess_x(container_x)
	container_y = _preprocess_y(container_y, num_classes)

	shard_filename = '{}-{}.tfrecords'.format(name, current_shard_id)
	current_shard_id += 1

	writer = tf.io.TFRecordWriter('{}/{}/{}'.format(FILE_PATH, data_dir, shard_filename))

	for image, label in zip(container_x, container_y):
		label_aux = np.array([label, label, label, label, label])
		example = tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature_jpeg(image), 'label': _bytes_feature(label.tostring()), 'label_aux': _bytes_feature(label_aux.tostring())}))
		writer.write(example.SerializeToString())

	writer.close()

def load_imagenet10(data_dir='imagenet10', only_test=False, only_bbox=False, aux_labels=False, batch_size=256):
	# 10 classes chosen from imagenet
	# Snake: n01742172 boa_constrictor
	# Dog: n02099712, Labrador_retriever 
	# Cat: n02123045, tabby
	# Frog: n01644373, tree_frog
	# Turtle: n01665541, leatherback_turtle
	# Bird: n01855672 goose
	# Bear: n02510455 giant_panda
	# Fish: n01484850 great_white_shark
	# Crab: n01981276 king_crab
	# Insect: n02206856 bee

	bbox_tag = '' if not only_bbox else '_bbox'
	train_cache_pattern = '{}/{}/{}-train-*'.format(FILE_PATH, data_dir, bbox_tag)
	test_cache_pattern = '{}/{}/{}-test-*'.format(FILE_PATH, data_dir, bbox_tag)

	#if cache does not exist, build
	if not (glob.glob(train_cache_pattern) and glob.glob(test_cache_pattern)):
		#load raw images (resized)
		train_data_gen, test_data_gen = _build_imagenet10(data_dir, size=320, only_test=only_test, only_bbox=only_bbox)

		_write_records(train_data_gen, 1024, data_dir, bbox_tag + '-train')
		_write_records(test_data_gen, 1024, data_dir, bbox_tag + '-test')

	#load cache as tf dataset
	train_tfrecords = tf.data.TFRecordDataset.list_files(train_cache_pattern, shuffle=True)
	test_tfrecords = tf.data.TFRecordDataset.list_files(test_cache_pattern, shuffle=False)

	train_dataset = tf.data.TFRecordDataset(train_tfrecords, num_parallel_reads=4)
	test_dataset = tf.data.TFRecordDataset(test_tfrecords, num_parallel_reads=4)

	_parse_function_filled = partial(_parse_function, aux_labels=aux_labels)
	train_dataset = train_dataset.map(map_func=_parse_function_filled, num_parallel_calls=4)
	test_dataset = test_dataset.map(map_func=_parse_function_filled, num_parallel_calls=4)

	train_dataset = train_dataset.map(map_func=_rebuild_image, num_parallel_calls=4)
	test_dataset = test_dataset.map(map_func=_rebuild_image, num_parallel_calls=4)

	train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
	train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False)
	train_dataset = train_dataset.prefetch(3)

	test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)

	if not only_test:
		train_dataset = train_dataset.repeat()
		test_dataset = test_dataset.repeat()

	if only_test:
		return None, test_dataset
	else:
		return train_dataset, test_dataset


def _build_imagenet10(data_dir, size, only_test, only_bbox):
	# builds and dumps imagenet10 to disk

	train_data_dir = '{}/{}/train'.format(FILE_PATH, data_dir)
	test_data_dir = '{}/{}/val'.format(FILE_PATH, data_dir)

	class_id_to_name = {}
	class_name_to_id = {}

	for class_id, class_name in enumerate(os.listdir(train_data_dir)):
		class_id_to_name[class_id] = class_name
		class_name_to_id[class_name] = class_id

	num_classes = len(class_id_to_name)

	#check number of classes
	assert(num_classes == 10)
	#check that the class names match between train, test and that they are in the map now
	assert(np.all([i in class_id_to_name.values() for i in np.unique(os.listdir(train_data_dir))]))
	assert(np.all([i in class_id_to_name.values() for i in np.unique(os.listdir(test_data_dir))]))

	if not only_test:
		train_data_gen = _load_images(train_data_dir, class_name_to_id, size, True)
	test_data_gen = _load_images(test_data_dir, class_name_to_id, size, False)
	return train_data_gen, test_data_gen

def _load_images(data_dir, class_name_to_id, size, bbox):
	# read and preprocess images from disk
	class_and_image_names = []

	for class_name in os.listdir(data_dir):
		for image in os.listdir(os.path.join(data_dir, class_name)):
			class_and_image_names.append((class_name, image))

	np.random.shuffle(class_and_image_names)

	for class_name, image in tqdm(class_and_image_names):
		file_dir = os.path.join(data_dir, class_name, image)

		#the image raw data
		image_data = tf.keras.preprocessing.image.load_img(file_dir, target_size=None, color_mode='rgb')
			
		if bbox:
			#crop to bounding box if exists, else ignore image
			coords = _bbox_coords(file_dir)
			if coords is None:
				#no bbox exists, skip
				continue
			else:
				#bbox exists, crop image
				xmin, xmax, ymin, ymax = coords
				image_data = image_data.crop((xmin, ymin, xmax, ymax)) 

		#the image label per assignments generated above
		image_label = class_name_to_id[class_name]
		
		#resize to 320, 320
		image_data = _resize_image(image_data, size)		     
		
		yield image_data, image_label

def _bbox_coords(image_file_dir):
	#returns bbox coords given image file if exists
    
	#check if bbox exists
	image_bbox_dir = image_file_dir.replace('train', 'bbox').replace('JPEG', 'xml')
	
	if not os.path.exists(image_bbox_dir):
		#none indicates no bbox file
		return None
		
	xmin = _parse_XML(image_bbox_dir, 'xmin')
	xmax = _parse_XML(image_bbox_dir, 'xmax')
	ymin = _parse_XML(image_bbox_dir, 'ymin')
	ymax = _parse_XML(image_bbox_dir, 'ymax')

	return xmin, xmax, ymin, ymax

def _parse_XML(xml_name, tag):
	#reads imagenet xml bbox for requested tag
	lines = [l for l in open(xml_name, 'r') if tag in l]
    
	#ensure only 1 bbox
	assert(len(lines) >= 0)
	#    if len(lines) > 1:
	#	    print('SKIPPING a BBOX')
	lines = lines[0]

	#assert(len(lines) == 1), '{}: {} | {}'.format(xml_name, len(lines), lines)
	#lines = lines[0]
	
	pattern = r"<{}>(.*?)</{}>".format(tag, tag)
	re_res = re.findall(pattern, lines)
	
	#ensure only 1 bbox
	assert(len(re_res) == 1), '{} @ {}: {} | {}'.format(xml_name, tag, len(re_res), re_res)
	re_res = int(re_res[0])
	
	return re_res

def _resize_image(image, size):
	# if image is smaller than largest glimpse, rescale to be as large as largest glimpse
	image_shape = np.array(image).shape
	enlarge_image_filter  = Image.BILINEAR
	aspect_ratio = image_shape[0] / image_shape[1]
	if image_shape[0] < size:
		height = size
		width = int((1. / aspect_ratio) * size)
		image = image.resize(size=(width, height), resample=enlarge_image_filter)
	image_shape = np.array(image).shape
	if image_shape[1] < size:
		width = size
		height = int(aspect_ratio * size)
		image = image.resize(size=(width, height), resample=enlarge_image_filter)
	image_shape = np.array(image).shape	

	glimpse_anchor_x = 0.5
	glimpse_anchor_y = 0.5
	
	glimpse_center_x = glimpse_anchor_x * image_shape[1]
	glimpse_center_y = glimpse_anchor_y * image_shape[0]
	glimpse_radius = size / 2.

	glimpse_xmin = glimpse_center_x - glimpse_radius
	glimpse_xmax = glimpse_center_x + glimpse_radius

	glimpse_ymin = glimpse_center_y - glimpse_radius
	glimpse_ymax = glimpse_center_y + glimpse_radius

	image = image.crop(box=(glimpse_xmin, glimpse_ymin, glimpse_xmax, glimpse_ymax))

	image = tf.keras.preprocessing.image.img_to_array(image)

	return image

def _preprocess_x(x):

	#rescale images from 0-255 to 0-1
	x = x.astype('float32') / 255.

	return x

def _preprocess_y(y, num_classes):

	#one hot encode y vector
	y = tf.keras.utils.to_categorical(y, num_classes)

	return y

def _invert_preprocess_y(y):

	#invert the preprocessing function

	y_invert = np.argmax(y, axis=-1)
	assert(len(y_invert)== len(y))

	return y_invert

def load_test10(batch_size, input_size):
	#for testing

	train_shape = (batch_size*3,input_size,input_size,3)
	test_shape  = (batch_size*2,input_size,input_size,3)

	x_train = np.ones(train_shape)
	y_train = np.zeros((train_shape[0], 10))
	y_train[:,0] = 1

	x_test = np.ones(test_shape)
	y_test = np.zeros((test_shape[0], 10))
	y_test[:,0] = 1

	return x_train, y_train, x_test, y_test

def load_cifar10(only_test=False):
	#cifar10 dataset

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

	#preprocess

	x_test = _preprocess_x(x_test)	
	y_test = _preprocess_y(y_test, 10)

	if only_test:
		x_train = None
		y_train = None
	else:
		x_train = _preprocess_x(x_train)
		y_train = _preprocess_y(y_train, 10)	

	return x_train, y_train, x_test, y_test

def load_cluttered_mnist(only_test=False):
	#cluttered mnist

	x_train = pickle.load(open('{}/cluttered_mnist/x_train.p'.format(FILE_PATH), 'rb'))
	x_test = pickle.load(open('{}/cluttered_mnist/x_test.p'.format(FILE_PATH), 'rb'))
	y_train = pickle.load(open('{}/cluttered_mnist/y_train.p'.format(FILE_PATH), 'rb'))
	y_test = pickle.load(open('{}/cluttered_mnist/y_test.p'.format(FILE_PATH), 'rb'))

	x_test = _preprocess_x(x_test)
	y_test = _preprocess_y(y_test, 10)

	if only_test:
		x_train = None
		y_train = None
	else:
		x_train = _preprocess_x(x_train)
		y_train = _preprocess_y(y_train, 10)

	return x_train, y_train, x_test, y_test

def load_integer_cifar10(only_test=False):
	#cifar10 dataset with image pixel values rounded to integers

	raise NotImplementedError

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

	#bin values
	x_train = np.round(x_train)
	x_test = np.round(x_test)

	#preprocess
	x_test = _preprocess_x(x_test)
	y_test = _preprocess_y(y_test, 10)

	if only_test:
		x_train = None
		y_train = None
	else:
		x_train = _preprocess_x(x_train)
		y_train = _preprocess_y(y_train, 10)

	return x_train, y_train, x_test, y_test

#def build_adv_dataset(model, name, random_relabel, x_train, y_train, cache=False, cache_file_x='', cache_file_y='', build_orthogonal_features=False):
def load_nonrobust_cifar10(model, name, random_relabel=True, cache=True, staggered_build=False, staggered_build_code=None, build_orthogonal_features=False):

	random_tag = 'random_relabel' if random_relabel else 'nonrandom_relabel'
	features_tag = 'multifeature_' if build_orthogonal_features else ''
	x_train_cache = '{}/cache_store/nonrobust_features_{}_{}_{}xtrain_adv.pickle'.format(FILE_PATH, name, random_tag, features_tag)
	y_train_cache = '{}/cache_store/nonrobust_features_{}_{}_{}ytrain_adv.pickle'.format(FILE_PATH, name, random_tag, features_tag)
	x_test_cache = '{}/cache_store/nonrobust_features_{}_{}_{}xtest_adv.pickle'.format(FILE_PATH, name, random_tag, features_tag)
	y_test_cache = '{}/cache_store/nonrobust_features_{}_{}_{}ytest_adv.pickle'.format(FILE_PATH, name, random_tag, features_tag)

	if build_orthogonal_features:
		perturbations_cache = '{}/cache_store/nonrobust_features_{}_{}_perturbations_adv.pickle'.format(FILE_PATH, name, random_tag)
		y_train_true_cache = '{}/cache_store/nonrobust_features_{}_{}_ytrain_true_adv.pickle'.format(FILE_PATH, name, random_tag)

	if cache:
		if os.path.exists(x_train_cache) and os.path.exists(y_train_cache) and os.path.exists(x_test_cache) and os.path.exists(y_test_cache):
			#if cache exists, just load from cache and return
			x_train_adv = pickle.load(open(x_train_cache, 'rb'))
			y_train_adv = pickle.load(open(y_train_cache, 'rb'))
			x_test = pickle.load(open(x_test_cache, 'rb'))
			y_test = pickle.load(open(y_test_cache, 'rb'))

			return x_train_adv, y_train_adv, x_test, y_test

	#else build nonrobust dataset, save to cache and return
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	y_train = y_train.flatten()
	y_test = y_test.flatten()

	#preprocess x
	x_train = _preprocess_x(x_train)
	x_test = _preprocess_x(x_test)

	#build nonrobust dataset	
	if random_relabel:
		#adv: rand()
		label_pool = np.arange(len(y_train)) % 10
		y_train_adv = np.random.choice(label_pool, replace=False, size=len(y_train)).astype('int64')
	else:
		#original: x_i	 y_i
		#adv:	   x_adv mod10(y_i+1)
		y_train_adv = np.mod(y_train + 1, 10).astype('int64')

	if build_orthogonal_features:
		x_train_adv, y_train_adv, perturbations, y_train_true = attack_backbone.build_nonrobust_features(model, x_train, y_train, y_train_adv, batch_size=500, build_orthogonal_features=build_orthogonal_features)

		y_train_true = _preprocess_y(y_train_true, 10)
		pickle.dump(perturbations, open(perturbations_cache, 'wb'))
		pickle.dump(y_train_true, open(y_train_true_cache, 'wb'))
	else:
		x_train_adv, y_train_adv = attack_backbone.build_nonrobust_features(model, x_train, y_train, y_train_adv, batch_size=500, build_orthogonal_features=build_orthogonal_features)

	#preprocess y
	y_train_adv = _preprocess_y(y_train_adv, 10)
	y_test = _preprocess_y(y_test, 10)

	#save to cache
	pickle.dump(x_train_adv, open(x_train_cache, 'wb'))
	pickle.dump(y_train_adv, open(y_train_cache, 'wb'))
	pickle.dump(x_test, open(x_test_cache, 'wb'))
	pickle.dump(y_test, open(y_test_cache, 'wb'))

	return x_train_adv, y_train_adv, x_test, y_test

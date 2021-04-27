#build models

import glimpse
import transformer

import tensorflow as tf
import tensorflow.keras.layers as layers

from functools import partial
from resnet_backbone import ResNet18
from resnet_backbone import ResNet_CIFAR
from resnet_backbone import resnet_layer

def resnet(input_shape=(320,320,3), base_model_input_shape=(224,224,3), name='CNN', num_classes=10, augment=False, sampling=False, coarse_fixations=True, coarse_fixations_upsample=False, coarse_fixations_gaussianblur=False, branched_network=False, gaze=None, return_logits=False, compat_sampling=False):
	#standard ImageNet models and derivatives 

	#check args
	if branched_network:
		if sampling:
			raise ValueError
		elif coarse_fixations:
			if coarse_fixations_upsample or coarse_fixations_gaussianblur:
				raise ValueError
		else:
			raise ValueError

	if sampling and coarse_fixations:
		raise NotImplementedError

	if coarse_fixations_upsample and (not coarse_fixations):
		raise ValueError

	if coarse_fixations_gaussianblur and (not coarse_fixations):
		raise ValueError

	if coarse_fixations_upsample and coarse_fixations_gaussianblur:
		raise ValueError

	if input_shape != (320, 320, 3):
		raise ValueError

	if coarse_fixations:
		if (not coarse_fixations_upsample) and (not coarse_fixations_gaussianblur):
			if base_model_input_shape != (224, 224, 3):
				raise ValueError
		elif coarse_fixations_upsample:
			if input_shape != base_model_input_shape:
				raise ValueError
		elif coarse_fixations_gaussianblur:
			if base_model_input_shape != (240, 240, 3):
				raise ValueError
		else:
			raise ValueError

	else:
		if input_shape != base_model_input_shape:
			raise ValueError

	#base model
	if not branched_network:
		network = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name=name, pooling='avg')
	else:
		network_branch1 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch1_{}'.format(name), pooling='avg', filters=27)
		network_branch2 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch2_{}'.format(name), pooling='avg', filters=27)
		network_branch3 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch3_{}'.format(name), pooling='avg', filters=27)
		network_branch4 = ResNet18(include_top=False, input_shape=base_model_input_shape, subnetwork_name='branch4_{}'.format(name), pooling='avg', filters=27)

	model_input = layers.Input(shape=input_shape)	

	#data augmentation
	if augment:
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='imagenet10'), name='image_augmentation')(model_input)
	else:
		x = model_input 

	#preprocess
	if coarse_fixations:

		if coarse_fixations_upsample:
			fixation_size = 160
		elif coarse_fixations_gaussianblur:
			fixation_size = 240
		else:
			fixation_size = 224

		if gaze is not None:
			coarse_foveation_x = tf.constant(gaze[0], tf.int32)
			coarse_foveation_y = tf.constant(gaze[1], tf.int32)
			coarse_fixation_center = [input_shape[0] // 2 + coarse_foveation_x, input_shape[0] // 2 + coarse_foveation_y]

			x = layers.Lambda(lambda tensor: glimpse.crop_square_patch(tensor, coarse_fixation_center, fixation_size), name='coarse_fixations')(x)
		else:
			x = layers.Lambda(lambda tensor: tf.image.random_crop(tensor, size=[tf.shape(tensor)[0], fixation_size, fixation_size, 3]), name='coarse_fixations')(x)


		if coarse_fixations_upsample:
			x = layers.Lambda(lambda tensor: glimpse.uniform_upsample(tensor, factor=2), name='uniform_upsampling')(x)
		if coarse_fixations_gaussianblur:
			x = layers.Lambda(lambda tensor: glimpse.gaussian_blur(tensor, radius=6), name='gaussian_blur')(x)

	if sampling:

		if gaze is not None:
			gaze_x = tf.constant(gaze[0], tf.int32)
			gaze_y = tf.constant(gaze[1], tf.int32)
			gaze = [gaze_x, gaze_y]
		else:
			#img shape (320, 320, 3)
			gaze = 80
	
		if compat_sampling:
			warp_image_filled = partial(glimpse.warp_image, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0], gaze=gaze)
			x = layers.Lambda(lambda tensor: tf.map_fn(warp_image_filled, tensor, back_prop=True), name='nonuniform_sampling')(x)
		else:
			x = layers.Lambda(lambda tensor: glimpse.warp_imagebatch(tensor, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0], gaze=gaze), name='nonuniform_sampling')(x)

	if not branched_network:
		x = network(x)
	else:
		x1 = network_branch1(x)
		x2 = network_branch2(x)
		x3 = network_branch3(x)
		x4 = network_branch4(x)
		x = layers.concatenate([x1, x2, x3, x4])

	if not return_logits:
		model_output = layers.Dense(num_classes, activation='softmax', name='probs')(x)
	else:
		model_output = layers.Dense(num_classes, activation=None, name='probs')(x)

	model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model

class GazeLayer(layers.Layer): # version 1: compute gaze using center of transformed image
	def __init__(self, input_shape, restricted_theta):
		super(GazeLayer, self).__init__()
		self.transformer = transformer.spatial_transformer_network
		self.shape = input_shape
		self.restricted = restricted_theta

	def call(self, inputs):
		return self.transformer(inputs[0], inputs[1], out_dims=[self.shape[0], self.shape[1]], restricted_theta=self.restricted)

class GazeLayer(layers.Layer): # version 2: compute gaze directly using a soft attention model (this is working)
	def __init__(self, input_shape, num_transformers):
		super(GazeLayer, self).__init__()
		self.attn_network = soft_attention_model(input_shape, 2, dummy_attention=False, dummy_scaled_attention=False, num_outputs=num_transformers)

	def call(self, inputs):
		return tf.transpose(self.attn_network(inputs))

class WarpLayer(layers.Layer):
	def __init__(self, input_shape):
		super(WarpLayer, self).__init__()
		self.warp = glimpse.warp_image_multi_gaze
		self.shape = input_shape

	def call(self, inputs):
		return self.warp(inputs[0], output_size=self.shape[0], input_size=self.shape[0], gaze=inputs[1])

def parallel_transformers(base_model_input_shape=(320,320,3), num_classes=10, return_logits=False, augment=False, restricted_attention=False, num_transformers=5, shared=False):
	if restricted_attention:
		num_theta_params = 4
	else:
		num_theta_params = 6

	model_input = layers.Input(shape=base_model_input_shape)
	
	if augment:
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='imagenet10'), name='image_augmentation')(model_input)
	else:
		x = model_input

	attn_network = soft_attention_model((base_model_input_shape), num_theta_params, dummy_attention=False,
						dummy_scaled_attention=False, num_outputs=num_transformers)
	theta = attn_network(x)

	# version 2
	gaze = GazeLayer(base_model_input_shape, num_transformers)(x)

	resnet_model = resnet(base_model_input_shape=base_model_input_shape, augment=False, coarse_fixations=False)
	resnet_model = tf.keras.models.Sequential(resnet_model.layers[:-1])

	x_transformed = [None]*num_transformers
	for i in range(num_transformers):
		# version 1
		# theta_i = theta[:, i*num_theta_params:(i+1)*num_theta_params]
		# _, _, gaze_i = GazeLayer(base_model_input_shape, restricted_attention)([x, theta_i])
		
		# version 2
		gaze_i = gaze[i*2:(i+1)*2, :]
		
		x_i = WarpLayer(base_model_input_shape)([x, gaze_i])

		if not shared:
			resnet_model = resnet(base_model_input_shape=base_model_input_shape, augment=False, coarse_fixations=False)
			resnet_model = tf.keras.models.Sequential(resnet_model.layers[:-1])

		x_transformed[i] = resnet_model(x_i)
	x = layers.concatenate(x_transformed)

	if not return_logits:
		model_output = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
	else:
		model_output = layers.Dense(num_classes, kernel_initializer='he_normal')(x)

	model = tf.keras.models.Model(inputs=model_input, outputs=model_output)
	
	return model
    
def ecnn(input_shape=(320,320,3), base_model_input_shape=(40,40,3), name='ECNN', num_classes=10, augment=False, auxiliary=False, sampling=True, scales='all', pooling=None, dropout=False, gaze=None, scale4_freeze=False, return_logits=False):
	#ImageNet cortical sampling model 

	#check args
	if input_shape != (320, 320, 3):
		raise ValueError

	if base_model_input_shape != (40, 40, 3):
		raise ValueError

	if scales not in ['all', 'scale4']:
		raise ValueError

	if scales != 'all' and auxiliary:
		raise ValueError

	if pooling is not None:
		if pooling not in ['max', 'avg']:
			raise ValueError

	if pooling is not None and dropout:
		raise NotImplementedError

	if scales == 'scale4':
		if dropout:
			raise NotImplementedError
		if pooling is not None:
			raise ValueError

	#base models
	if scales == 'all':
		scale1_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale1-{}'.format(name), pooling='avg')	
		scale2_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale2-{}'.format(name), pooling='avg')
		scale3_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale3-{}'.format(name), pooling='avg')
	scale4_network = ResNet18(include_top=False, input_shape=base_model_input_shape, conv1_stride=1, max_pool_stride=1, filters=45, subnetwork_name='scale4-{}'.format(name), pooling='avg')		
	
	model_input = layers.Input(shape=input_shape)

	#data augmentation
	if augment:
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='imagenet10'), name='image_augmentation')(model_input)
	else:
		x = model_input 

	#preprocess
	if gaze is not None:
		gaze_x = tf.constant(gaze[0], tf.int32)
		gaze_y = tf.constant(gaze[1], tf.int32)
		gaze = [gaze_x, gaze_y]
	else:
		#img shape (320, 320, 3)
		if not scale4_freeze:
			gaze = 40
		else:
			gaze = 80

	if not scale4_freeze:
		scale_sizes = [40, 80, 160, 240]
		scale_radii = [1, 2, 4, 6]
	else:
		scale_sizes = [40, 80, 160, 320]
		scale_radii = [1, 2, 4, 8]
		
	scale_center = [input_shape[0] // 2, input_shape[0] // 2]

	if not sampling:
		scales_x = layers.Lambda(lambda tensor: glimpse.image_scales(tensor, scale_center, scale_radii, scale_sizes, gaze, scale4_freeze), name='scale_sampling')(x)
	else:
		scales_x = layers.Lambda(lambda tensor: glimpse.warp_image_and_image_scales(tensor, input_shape[0], input_shape[0], scale_center, scale_radii, scale_sizes, gaze, scale4_freeze), name='nonuniform_and_scale_sampling')(x)
		
	#unpack scales
	scale1_x = scales_x[0]
	scale2_x = scales_x[1]
	scale3_x = scales_x[2]
	scale4_x = scales_x[3]

	if scales == 'all':
		scale1_x = scale1_network(scale1_x)
		scale2_x = scale2_network(scale2_x)
		scale3_x = scale3_network(scale3_x)
	scale4_x = scale4_network(scale4_x)

	if scales == 'all':
		if pooling is None:
			x = layers.concatenate([scale1_x, scale2_x, scale3_x, scale4_x])
		elif pooling == 'avg':
			x = layers.Average()([scale1_x, scale2_x, scale3_x, scale4_x])
		elif pooling == 'max':
			x = layers.Maximum()([scale1_x, scale2_x, scale3_x, scale4_x])
		else:
			raise ValueError

		if dropout:
			x = layers.Dropout(0.75)(x)
	elif scales == 'scale4':
		x = scale4_x
	else:
		raise ValueError

	if not return_logits:
		model_output = layers.Dense(num_classes, activation='softmax', name='probs')(x)
	else:
		model_output = layers.Dense(num_classes, activation=None, name='probs')(x)

	if auxiliary:
		#aux output

		if return_logits:
			raise NotImplementedError

		scale1_aux_out = layers.Dense(num_classes, activation='softmax', name='scale1_aux_probs')(scale1_x)
		scale2_aux_out = layers.Dense(num_classes, activation='softmax', name='scale2_aux_probs')(scale2_x)
		scale3_aux_out = layers.Dense(num_classes, activation='softmax', name='scale3_aux_probs')(scale3_x)
		scale4_aux_out = layers.Dense(num_classes, activation='softmax', name='scale4_aux_probs')(scale4_x)

		model = tf.keras.models.Model(inputs=model_input, outputs=[model_output, scale1_aux_out, scale2_aux_out, scale3_aux_out, scale4_aux_out])
	else:
		model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model

def resnet_cifar(input_shape=(32,32,3), base_model_input_shape=(24,24,3), name=None, num_classes=10, augment=False, sampling=False, coarse_fixations=True, coarse_fixations_upsample=False, coarse_fixations_gaussianblur=False, gaze=None, return_logits=False, approx_ecnn=False, compat_sampling=False, build_feedback=False, ecnn_pooling=None, ecnn_dropout=False, ecnn_single_scale=False, adv_training=False, wider_network=False):
	#all resnet architectures for cifar10
	
	#check args
	if adv_training:
		assert(gaze is None)	
    
	if name is not None:
		raise NotImplementedError

	if build_feedback:
		if coarse_fixations or sampling or approx_ecnn or coarse_fixations_upsample:
			raise NotImplementedError

	if coarse_fixations and sampling:
		raise NotImplementedError

	if approx_ecnn and sampling:
		raise ValueError
	if approx_ecnn and coarse_fixations:
		raise ValueError

	if ecnn_single_scale:
		assert(approx_ecnn)
		assert(ecnn_pooling is None)
		assert(not ecnn_dropout)

	if ecnn_pooling is not None:
		assert(approx_ecnn)
		if ecnn_pooling not in ['max', 'avg']:
			raise ValueError

	if ecnn_pooling is not None and ecnn_dropout:
		raise NotImplementedError
	
	if ecnn_dropout:
		assert(approx_ecnn)
	
	if not coarse_fixations and coarse_fixations_upsample:
		raise ValueError
	
	if not coarse_fixations and coarse_fixations_gaussianblur:
		raise ValueError

	if input_shape != (32, 32, 3):
		raise ValueError
	if coarse_fixations:
		if not coarse_fixations_upsample:
			if base_model_input_shape != (24, 24, 3):
				raise ValueError
		else:
			if base_model_input_shape != (32, 32, 3):
				raise ValueError
	else:
		if approx_ecnn:
			if base_model_input_shape != (12, 12, 3):
				raise ValueError
		else:
			if base_model_input_shape != (32, 32, 3):
				raise ValueError

	if input_shape != base_model_input_shape and (not coarse_fixations) and (not approx_ecnn):
		raise ValueError

	#base model
	if not approx_ecnn:
		if not wider_network:
			network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits, build_feedback=build_feedback)
		else:
			network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits, build_feedback=build_feedback, num_filters=32)
	else:
		if not wider_network:
			num_filters_per_scale=21
		else:
			num_filters_per_scale=43
		if not ecnn_single_scale:	 
			scale1_network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits, num_filters=num_filters_per_scale, return_latent=True, skip_last_downsample=True)
		scale2_network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=return_logits, num_filters=num_filters_per_scale, return_latent=True, skip_last_downsample=True)

	if adv_training:
		gaze_input = layers.Input(shape=(2), dtype=tf.int32)
	model_input = layers.Input(shape=input_shape)	

	#data augmentation
	if augment:
		assert(not adv_training)	
		x = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='cifar10'), name='image_augmentation')(model_input)
	else:
		if adv_training:
			gaze = gaze_input
		x = model_input

	#preprocess
	if coarse_fixations:

		if not coarse_fixations_upsample:
			fixation_size = 24
		else:
			fixation_size = 16

		if gaze is not None:
			if not adv_training:		
				assert(isinstance(gaze, list))
				coarse_foveation_x = tf.constant(gaze[0], tf.int32)
				coarse_foveation_y = tf.constant(gaze[1], tf.int32)
				coarse_fixation_center = [input_shape[0] // 2 + coarse_foveation_x, input_shape[0] // 2 + coarse_foveation_y]
			else:
				gaze = gaze + (input_shape[0] // 2)

			if not adv_training:		
				x = layers.Lambda(lambda tensor: glimpse.crop_square_patch(tensor, coarse_fixation_center, fixation_size), name='coarse_fixations')(x)
			else:
				x = layers.Lambda(lambda tensor: glimpse.crop_square_patch_wrapper(tensor, patch_size=fixation_size), name='coarse_fixations')([x, gaze])
	    
		else:
			x = layers.Lambda(lambda tensor: tf.image.random_crop(tensor, size=[tf.shape(tensor)[0], fixation_size, fixation_size, 3]), name='coarse_fixations')(x)
			
		if coarse_fixations_upsample:
			x = layers.Lambda(lambda tensor: glimpse.uniform_upsample(tensor, factor=2), name='uniform_upsampling')(x)
			
		if coarse_fixations_gaussianblur:
			x = layers.Lambda(lambda tensor: glimpse.gaussian_blur(tensor, radius=2), name='gaussian_blur')(x)

	if sampling:
		if gaze is not None:
			if not adv_training:		
				assert(isinstance(gaze, list))
				gaze_x = tf.constant(gaze[0], tf.int32)
				gaze_y = tf.constant(gaze[1], tf.int32)
				gaze = [gaze_x, gaze_y]
		else:
			#img shape (32, 32, 3)
			gaze = 8

		if compat_sampling:
			assert(not adv_training)
			warp_image_filled = partial(glimpse.warp_image, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0], gaze=gaze)
			x = layers.Lambda(lambda tensor: tf.map_fn(warp_image_filled, tensor, back_prop=True), name='nonuniform_sampling')(x)
		else:
			if adv_training:	    
				x = layers.Lambda(lambda tensor: glimpse.warp_imagebatch_wrapper(tensor, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0]), name='nonuniform_sampling')([x, gaze])
			else:		 
				x = layers.Lambda(lambda tensor: glimpse.warp_imagebatch(tensor, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0], gaze=gaze), name='nonuniform_sampling')(x)
	    
	if approx_ecnn:
		if gaze is not None:
			if not adv_training:
				assert(isinstance(gaze, list))
				gaze_x = tf.constant(gaze[0], tf.int32)
				gaze_y = tf.constant(gaze[1], tf.int32)
				gaze = [gaze_x, gaze_y]
		else:
			#larger crop is (30, 30, 3)
			gaze = 4

		scale_sizes = [12, 24]
		scale_radii = [1, 2]

		scale_center = [input_shape[0] // 2, input_shape[0] // 2]
	
		if not adv_training:	   
			scales_x = layers.Lambda(lambda tensor: glimpse.image_scales_CIFAR(tensor, scale_center, scale_radii, scale_sizes, gaze), name='scale_sampling')(x)
		else:
			scales_x = layers.Lambda(lambda tensor: glimpse.image_scales_CIFAR_wrapper(tensor, scale_center=scale_center, scale_radii=scale_radii, scale_sizes=scale_sizes), name='scale_sampling')([x, gaze])

		scale1_x = scales_x[0]
		scale2_x = scales_x[1]

	if not approx_ecnn:
		model_output = network(x)		
	else:
		if ecnn_single_scale:
			x = scale2_network(scale2_x)		
		else:		 
			scale1_x = scale1_network(scale1_x)
			scale2_x = scale2_network(scale2_x)
			if ecnn_pooling is None:
				x = layers.concatenate([scale1_x, scale2_x])
			elif ecnn_pooling == 'avg':
				x = layers.Average()([scale1_x, scale2_x])
			elif ecnn_pooling == 'max':
				x = layers.Maximum()([scale1_x, scale2_x])
			else:
				raise ValueError

			if ecnn_dropout:
				x = layers.Dropout(0.5)(x)	  
	
		if not return_logits:
			model_output = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
		else:
			model_output = layers.Dense(num_classes, kernel_initializer='he_normal')(x)

	if adv_training:
		model = tf.keras.models.Model(inputs=[model_input, gaze_input], outputs=model_output)
	else:
		model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model

def attention_mnist(augment=False, return_logits=True, attention=False, dummy_attention=False, dummy_scaled_attention=False, restricted_attention=False, retinal_attention=False):
	#architecture for attention mechanisms on mnist

	if dummy_attention:
		assert(attention)
		assert(not dummy_scaled_attention)
		assert(not restricted_attention)
		assert(not retinal_attention)
	
	if dummy_scaled_attention:
		assert(attention)
		assert(not dummy_attention)
		assert(not restricted_attention)
		assert(not retinal_attention)

	if restricted_attention:
		assert(attention)
		assert(not dummy_attention)
		assert(not dummy_scaled_attention)
		assert(not retinal_attention)
	
	if retinal_attention:
		assert(attention)
		assert(not dummy_attention)
		assert(not dummy_scaled_attention)
		assert(not restricted_attention)
		gaze_scale=20

	base_model_input_shape = (40,40,1)
	num_classes=10
	if restricted_attention:
		num_coords = 4
	elif retinal_attention:
		num_coords = 2
	else:
		num_coords = 6

	model_input = layers.Input(shape=base_model_input_shape)

	if augment:
		x_input = layers.Lambda(lambda tensor: glimpse.image_augmentation(tensor, dataset='cluttered_mnist'), name='image_augmentation')(model_input)
	else:
		x_input = model_input 

	if attention:
		attention_network = soft_attention_model((base_model_input_shape), num_coords, dummy_attention, dummy_scaled_attention)
		gaze = attention_network(x_input)
		if not retinal_attention:
			x = layers.Lambda(lambda tensor: transformer.spatial_transformer_network(tensor[0], tensor[1], out_dims=[base_model_input_shape[0], base_model_input_shape[1]], restricted_theta=restricted_attention), name='transformer')([x_input, gaze])
		else:
			gaze = gaze*gaze_scale	  
			warp_image_filled = partial(glimpse.warp_image_mapsupported, output_size=base_model_input_shape[0], input_size=base_model_input_shape[0])
			x = layers.Lambda(lambda tensor: tf.map_fn(warp_image_filled, (tensor[0], tensor[1]), dtype=tf.float32, back_prop=True), name='nonuniform_sampling')([x_input, gaze])
    
	else:
		x = x_input
	
	#network = ResNet_CIFAR(n=3, version=1, input_shape=base_model_input_shape, num_classes=num_classes, verbose=0, return_logits=False, build_feedback=False, return_latent=True, pool_latent=True)
	#x = network(x)

	x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(x)
	x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(1024, activation='relu')(x)
	x = layers.Dropout(0.2)(x)

	if not return_logits:
		model_output = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
	else:
		model_output = layers.Dense(num_classes, kernel_initializer='he_normal')(x)

	model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

	return model

def soft_attention_model(input_shape, num_coords, dummy_attention, dummy_scaled_attention, num_outputs=1):
	#build attention network
	
	assert(type(input_shape) == tuple)
	if num_coords != 6 and num_coords != 4 and num_coords != 2:
		raise NotImplementedError

	resnet_model = resnet(base_model_input_shape=input_shape, augment=False, coarse_fixations=False)
	attention_network = tf.keras.models.Sequential(resnet_model.layers[:-1])

	# attention_network = ResNet_CIFAR(n=3, version=1, input_shape=input_shape, num_classes=-1, verbose=0, return_logits=False, return_latent=True, build_feedback=False, skip_downsamples=False)

	# attn_in = layers.Input(shape=input_shape)

	# attn_x = layers.Flatten()(attn_in)
	# attn_x = layers.Dense(50, activation='tanh')(attn_x)
	# attn_out = layers.Dropout(0.2)(attn_x)

	# attention_network = tf.keras.models.Model(inputs=attn_in, outputs=attn_out)

	model_input = layers.Input(shape=input_shape)
	x = attention_network(model_input)

	if num_coords == 6:    
		if dummy_scaled_attention:
			bias_init = [0.75, 0., 0., 0., 0.75, 0.]
		else:
			bias_init = [1., 0., 0., 0., 1., 0.]
	elif num_coords == 4:
		bias_init = [1., 0., 1., 0.]
	elif num_coords == 2:
		bias_init = [0., 0.]
    	
	bias_init = [val for _ in range(num_outputs) for val in bias_init]
	
	model_output = layers.Dense(num_coords * num_outputs, activation='tanh', kernel_initializer=tf.zeros_initializer(), bias_initializer=tf.constant_initializer(bias_init))(x)

	attn_model = tf.keras.models.Model(inputs=model_input, outputs=model_output)
	if dummy_attention or dummy_scaled_attention:
		attn_model.trainable = False

	return attn_model


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Activation, concatenate, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


class L2Layer(Layer):
	def __init__(self, **kwargs):
		super(L2Layer, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self._x = self.add_weight(name='alpha_l2', 
									shape=(1,),
									initializer='ones',
									trainable=True)
		super(L2Layer, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		return self._x * tf.divide(x, tf.norm(x, ord='euclidean'))

	def compute_output_shape(self, input_shape):
		return input_shape[0]
		
	# Fix me: Rise an error at serialization time: 'Not JSON Serializable...' apparently is a TF error.
	'''
	def get_config(self):
		base_config = super(L2Layer, self).get_config()
		config = {'alpha_l2': self._x}
		return config
	'''




def pretrainedmodel(classes, name_model='VGG16', use_l2=True, is_trainable=False):
	models = ['VGG16', 'ResNet50', 'InceptionV3', 'Xception']

	if name_model not in models:
		print('Name model not found. Try {}'.format(models))
		return 

	if name_model == 'VGG16':
		expected_dim = (224, 224, 3)
		vgg16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
		basemodel = Model(inputs=vgg16.inputs, outputs=vgg16.get_layer(index=-1).output)

	if name_model == 'ResNet50':
		expected_dim = (224, 224, 3)
		resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
		basemodel = Model(inputs=resnet.inputs, outputs=resnet.get_layer(index=-1).output)

	if name_model == 'InceptionV3':
		expected_dim = (299, 299, 3)
		iv3 = InceptionV3(weights='imagenet',include_top=False, pooling='avg')
		basemodel = Model(inputs=iv3.inputs, outputs=iv3.get_layer(index=-1).output)

	if name_model == 'Xception':
		expected_dim = (299, 299, 3)
		xcep = Xception(weights='imagenet', include_top=False, pooling='avg')
		basemodel = Model(inputs=xcep.inputs, outputs=xcep.get_layer(index=-1).output)


	# freeze weights
	basemodel.trainable = is_trainable
	x = basemodel.output

	# add l2-norm constraint
	if use_l2:
		x = L2Layer()(x)

    # predictions
	y = Dense(classes, activation='softmax', name='predictions')(x)
	model = Model(inputs=basemodel.input, outputs=y)

	return model, expected_dim
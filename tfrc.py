import os
import tensorflow as tf
from sys import argv
import random
import numpy as np


def write_tfrecords(data, writer):
	new_data = data.tolist()
	
	example = tf.train.Example(features=tf.train.Features(feature={
		#"x": tf.train.Feature(int64_list=tf.train.Int64List(value=data)),
		"x": tf.train.Feature(float_list=tf.train.FloatList(value=new_data)),
		}))

	writer.write(example.SerializeToString())


def get_input_data(input_file, seq_length, batch_size, buffer_size=10):
	def parser(record):
			name_to_features = {
					"x": tf.FixedLenFeature([seq_length], tf.int64),
					#"x": tf.FixedLenFeature([seq_length], tf.float32),
			}

			example = tf.parse_single_example(record, features=name_to_features)
			#input_ids = tf.cast(example["x"], tf.float32)
			input_ids = example["x"]

			return input_ids
	
	dataset = tf.data.TFRecordDataset(input_file)
	dataset = dataset.map(parser).repeat().batch(batch_size)
	dataset = dataset.shuffle(buffer_size=buffer_size)
	iterator = dataset.make_one_shot_iterator()
	input_ids = iterator.get_next()
	
	return input_ids



if __name__ == '__main__':
	path = "/data/ai-datasets/201-FFHQ/tfrecords/face1024.tfrecords"
	data, labels = get_input_data(path, 1024*1024*3, 2)
	sess = tf.Session()
	for i in range(10):
		print(sess.run(data))
	#print(sess.run(labels))
	sess.close()



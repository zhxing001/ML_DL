
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
	'a named tuple describing a resnet block'

def subsample(inputs,factor,scope=None):
	if factor==1:
		return inputs
	else:
		return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
	if stride==1:
		return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME',scope=scope)
	else:
		pad_total=kernel_size-1
		pad_beg=pad_total//2
		pad_end=pad_total-pad_beg
		inputs=tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
		return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding='VALID',scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):
	for block in blocks:
		with tf.variable_scope(block.scope,'block',[net]) as sc:
			for i,unit in enumerate(block.args):
				with tf.variable_scope('unit_%d'%(i+1),values=[net]):
					unit_depth,unit_depth_bottleneck,unit_stride=unit
					net=block.unit_fn(net,
									  depth=unit_depth,
									  depth_bottleneck=unit_depth_bottleneck,
									  stride=unit_stride)
				net=slim.utils.collec_named_outputs(outputs_collections,sc.name,net)
	return net


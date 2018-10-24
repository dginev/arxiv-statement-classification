import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 16
config.inter_op_parallelism_threads = 16

f = open("config.bin", "wb")
f.write(config.SerializeToString())
f.close()

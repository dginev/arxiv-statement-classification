import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16, device_count={
                        "CPU": 16}, allow_soft_placement=True)

f = open("config.bin", "wb")
f.write(config.SerializeToString())
f.close()

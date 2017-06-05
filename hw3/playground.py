import tensorflow as tf


sess = tf.Session()
a = tf.placeholder(tf.float32,shape=[None,6])
b = tf.placeholder(tf.float32,shape=[None,6])
adder_nodes = a + b
q_val = tf.reduce_sum(a*b,axis=1)

#print(sess.run(adder_nodes,{a:[[1,1,1,1,1,1],[2,1,1,1,1,1]],b:[[1,1,1,1,1,1],[2,1,1,1,1,1]]}))
print(sess.run(a*b,{a:[[1,1,1,1,1,1],[2,1,1,1,1,1]],b:[[1,1,1,1,1,1],[2,1,1,1,1,1]]}))
print(sess.run(q_val,{a:[[1,1,1,1,1,1],[2,1,1,1,1,1]],b:[[1,1,1,1,1,1],[2,1,1,1,1,1]]}))
#print(sess.run(adder_nodes,{a:[1,3],b:[2,4]}))

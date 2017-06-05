import tensorflow as tf


#matrix1 = tf.constant([[3., 3.]])
#matrix2 = tf.constant([[2.],[2.]])
#product = tf.matmul(matrix1, matrix2)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1,node2)
print('node3 = ',node3)
print('sess.run(node3)',sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_nodes = a + b

print(sess.run(adder_nodes,{a:3,b:4.5}))
print(sess.run(adder_nodes,{a:[1,3],b:[2,4]}))

#---------
# y = Wx+b

W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model,{x:[1,2,3,4]}))	


y = tf.placeholder(tf.float32)
sq_diff = tf.square(linear_model - y)
loss = tf.reduce_sum(sq_diff)
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

fixW = tf.assign(W,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
   sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})


print(sess.run([W,b]))




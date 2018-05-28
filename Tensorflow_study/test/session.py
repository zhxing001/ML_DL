import tensorflow as tf

mat1=tf.constant([[2,2]])
mat2=tf.constant([[1],[5]])
product=tf.matmul(mat1,mat2)

###
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()
###




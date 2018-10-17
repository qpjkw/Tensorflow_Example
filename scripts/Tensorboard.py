
# coding: utf-8

# In[1]:


"""
Quickstart to Tensorboard
The dataset is using MNIST handwritten dataset.

author: Jiankai Wang
"""


# In[2]:


from __future__ import print_function
import os
import tensorflow as tf
print("Tensorflow Version: {}".format(tf.__version__))


# # Prepare

# ## Load MNIST dataset

# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/mnist_data", one_hot=True)


# # Parameters / Model

# ## hyper-parameters

# In[4]:


learning_rate = 1e-2
training_epochs = 100
batch_size = 1000
display_step = 1
log_path = os.path.join("/","tmp","tensorboard_log","mnist_example")


# ## network

# In[5]:


n_input = 784      # = 28 (height) * 28 (width)
n_hidden_1 = 512   # the neural number in 1st hidden layer
n_hidden_2 = 256   # the neural number in 2nd hidden layer
n_output = 10      # the output / classification number


# In[6]:


def layer(input, weight_shape, bias_shape):
    weight_std = (2.0 / weight_shape[0]) ** 0.5                              # weight normalization
    w_init = tf.random_normal_initializer(stddev=weight_std)                 # normalize the weight parameters
    b_init = tf.constant_initializer(value=0)
    W = tf.get_variable(name="W", shape=weight_shape, initializer=w_init)
    b = tf.get_variable(name="b", shape=bias_shape, initializer=b_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


# In[7]:


def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [n_input, n_hidden_1], [n_hidden_1])
        
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
    
    with tf.variable_scope("output"):
        output = layer(hidden_2, [n_hidden_2, n_output], [n_output])
        
    return output


# # Learning

# ## Target

# In[8]:


def loss(output, y):
    """
    output: the logits value from inference
    y: the labeling data
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)
    loss = tf.reduce_mean(cross_entropy)
    return loss


# In[9]:


def training(loss, global_step):
    """
    loss: the loss value
    global_step: the global training step index
    """
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)
    return grads, apply_grads


# In[10]:


def evaluate(output, y):
    """
    output: the logits value from inference
    y: the labeling data
    """
    compare = tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))
    tf.summary.scalar("eval", accuracy)
    return accuracy


# ## Training

# In[11]:


print("""
Run 'tensorboard --logdir=/tmp/tensorboard_log/' to monitor the training process.
""")


# In[12]:


with tf.Graph().as_default():
    with tf.variable_scope("mlp"):
        x = tf.placeholder("float", [None, 784])  # x is batch input
        y = tf.placeholder("float", [None, 10])   # y is output for 10 classification

        output = inference(x)   # get the inference result
        loss_val = loss(output=output, y=y)   # get the loss
        global_step = tf.Variable(0, name="global_step", trainable=False)   # training step
        train_grads, train_opt = training(loss=loss_val, global_step=global_step)   # training body
        eval_opt = evaluate(output=output, y=y)   # evaluation result
        
        # show all training variable info
        # may cause summary name error
        # INFO:tensorflow:Summary name mlp/hidden_1/W:0 is illegal; using mlp/hidden_1/W_0 instead.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
            
        # show grads info
        for grad, var in train_grads:
            tf.summary.histogram(var.name + '/gradient', grad)

        init_var = tf.global_variables_initializer()
        summary_opt = tf.summary.merge_all()   # merge all summaries
        saver = tf.train.Saver()   # for saving checkpoints

        with tf.Session() as sess:
            
            summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)   # write the summary 
            sess.run(init_var)   # initialize all variables

            for epoch in range(training_epochs):
                avg_loss = 0.
                total_batch = int(mnist.train.num_examples / batch_size)

                for idx in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)   # get the batch data

                    feed_dict_data = {x: batch_x, y: batch_y}
                    grads, _ = sess.run([train_grads, train_opt], feed_dict=feed_dict_data)   # run training

                    batch_loss = sess.run(loss_val, feed_dict=feed_dict_data)
                    avg_loss += batch_loss / total_batch   # calculate the average loss

                if epoch % display_step == 0:
                    # record log

                    feed_dict_val_data = {x: mnist.validation.images, y: mnist.validation.labels}
                    acc = sess.run(eval_opt, feed_dict=feed_dict_val_data)   # calculate the accuracy

                    print("Epoch: {}, Accuracy: {}, Vaildation Error: {}".format(epoch+1, round(acc,2), round(1-acc,2)))
                    tf.summary.scalar("validation_accuracy", acc)  

                    summary_str = sess.run(summary_opt, feed_dict=feed_dict_val_data)
                    summary_writer.add_summary(summary_str, sess.run(global_step))   # write out the summary

                    saver.save(sess, os.path.join(log_path, "model-checkpoint"), global_step=global_step)

            print("Training finishing.")

            feed_dict_test_data = {x: mnist.test.images, y: mnist.test.labels}
            acc = sess.run(eval_opt, feed_dict=feed_dict_test_data)   # test result
            print("Test Accuracy:",acc)


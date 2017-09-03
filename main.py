import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    
    vgg_tag = ['vgg16']
    
    # returns the MetaGraphDef protocol buffer loaded in the provided session. 
    # This can be used to further extract signature-defs, collection-defs, etc.
    tf.saved_model.loader.load(sess,vgg_tag,vgg_path)
    
    
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    graph = tf.get_default_graph()
    
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

# perform test to verify load_vgg function implementation
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    #print ("num_classes=",num_classes) #debug
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Following same architecture of the paper
    # First: we apply the 1st techinique to add a 1x1 Convolutional layer to the end of the original CNN 
    # layer8 name = layer_1x1
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out,num_classes,kernel_size=1,strides=1,padding='same',name='conv_1x1',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Second: we apply transformations to upsample or deconvolute the layers
    # Upsample by 2
    updeconv2 = tf.layers.conv2d_transpose(conv_1x1,num_classes,kernel_size=4,strides=2,padding='same',name='updeconv2',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out,num_classes,kernel_size=1,strides=1,padding='same',name='layer4_1x1',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    skip_4 = tf.add(updeconv2,layer4_1x1)
    # Upsample by 2
    updeconv4 = tf.layers.conv2d_transpose(skip_4,num_classes,kernel_size=4,strides=2,padding='same',name='updeconv4',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out,num_classes,kernel_size=1,strides=1,padding='same',name='layer3_1x1',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #up_layer4= tf.layers.conv2d_transpose(layer4_1x1,num_classes,kernel_size=4,strides=2,padding='same',name='up_layer4',
#                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip_3 = tf.add(updeconv4,layer3_1x1)
    #skip_3_4 = tf.add(skip_3,up_layer4)

    # Upsample by 8
    #updeconv32 = tf.layers.conv2d_transpose(skip_3_4,num_classes,kernel_size=16,strides=8,padding='same',name='updeconv32',
     #                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    updeconv32 = tf.layers.conv2d_transpose(skip_3,num_classes,kernel_size=16,strides=8,padding='same',name='updeconv32',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return updeconv32

#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    my_labels = tf.reshape(correct_label, (-1, num_classes))
    my_logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # calculate loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=my_logits, labels=my_labels))
    # adam optimizer
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return my_logits, train_op, cross_entropy_loss

#tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    #my_learning_rate = tf.placeholder(dtype=tf.float32)
    my_learning_rate = learning_rate

    #my_keep_prob = tf.placeholder(dtype=tf.float32)
    my_keep_prob = keep_prob

    count = 0
    for epoch in range(epochs):

        for (image, my_label) in get_batches_fn(batch_size):
            discard, loss = sess.run([train_op, cross_entropy_loss],
                 feed_dict={input_image:image, correct_label:my_label, keep_prob:my_keep_prob,learning_rate:my_learning_rate})
        print("Iter=",str(count)," Epoch=", str(epoch), "/", str(epochs), " loss=", str(loss))
        #print('Epoch={}/{} count={} loss={}'.format(epoch, epochs, count, loss))
        count = count + 1
            
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 50   #50
    batch_size = 30  #10

    learning_rate = tf.placeholder(dtype=tf.float32)
    learning_rate = tf.Variable(0.00001)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg') #OK
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape) #OK

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path) #OK

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
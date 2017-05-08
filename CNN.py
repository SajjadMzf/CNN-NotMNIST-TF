# Import
#
import tensorflow as tf
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time
import progressbar
do_train = 0
do_test = 0
do_userTest = 1
start = time.time()
dataset_train_name = "notMNIST_large.tar.gz"
dataset_test_name = "notMNIST_small.tar.gz"
model_dir= "./Model/"
user_test_dir ="./user_test/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_name = "CNN.ckpt"
target_dir = "./Dataset/"
NUM_THREADS = 8
device_type = "/gpu:0"


# Load Dataset
#
print('Loading Dataset...')
def extract_dataset(dir, dataset_file):
    dataset_dir = os.path.join(dir,dataset_file)
    if not os.path.exists(dataset_dir):
        print(dataset_file, "not found!")
        return
    dataset_folder = os.path.splitext(os.path.splitext(dataset_file)[0])[0]
    if os.path.exists(os.path.join(dir, dataset_folder)):
        print("Already extracted!")
        return
    tar = tarfile.open(dataset_dir)
    tar.extractall(dir)
    tar.close()
    print(dataset_file, "extracted!")

extract_dataset(target_dir, dataset_train_name)
extract_dataset(target_dir, dataset_test_name)

image_size = 28

def load_class(dir,class_folder):
    class_dir = os.path.join(dir, class_folder)
    if not os.path.exists(class_dir):
        print(class_folder, "not found!")
        return
    image_files = os.listdir(class_dir)
    class_dataset = np.ndarray(shape=(len(image_files), image_size, image_size))
    image_num = 0
    for image in image_files:
        image_dir = os.path.join(class_dir,image)
        try:
            image_data = ndimage.imread(image_dir).astype(float)
            if image_data.shape !=(image_size, image_size):
                print('Unexpected image size:', image, '-skipping')
            class_dataset[image_num, :, :] = image_data
            image_num = image_num + 1
        except IOError as e:
            print(e, '-skipping')
    return class_dataset, image_num

def load_dataset(dir, dataset_folder, shuffle = True):
    dataset_dir = os.path.join(dir, dataset_folder)
    if not os.path.exists(dataset_dir):
        print(dataset_folder, "not found!")
    class_files = sorted(os.listdir(dataset_dir))
    class_num = len(class_files)
    dataset = {}
    dataset['input'] = []
    dataset['target'] = []
    dataset_size = 0
    for idx, file in enumerate(class_files):
        class_data, class_size = load_class(dataset_dir,file)
        dataset['input'].extend(class_data)
        dataset['target'].extend((idx)*np.ones(shape = class_size))
        dataset_size = dataset_size + class_size
    if shuffle:
        combined = list(zip(dataset['input'],dataset['target']))
        np.random.shuffle(combined)
        dataset['input'], dataset['target'] = zip(*combined)
    return dataset, class_num, dataset_size


def user_data_loader(data_dir):
    if not os.path.exists(data_dir):
        return 0,0
    user_data_file = sorted(os.listdir(data_dir))
    data_size = 0
    user_dataset = np.ndarray(shape=(len(user_data_file), image_size, image_size))
    user_data_name =[]
    for image in user_data_file:
        image_dir = os.path.join(data_dir,image)
        try:
            image_data = ndimage.imread(image_dir, flatten=True).astype(float)
            if image_data.shape !=(image_size, image_size):
                print('Unexpected image size:', image, image_data.shape, '-skipping')
            else:
                user_dataset[data_size, :, :] = image_data
                user_data_name.append(image)
                data_size = data_size + 1
        except IOError as e:
            print(e, '-skipping')
    return user_dataset,user_data_name, data_size



if do_test == 1:
    dataset_test_folder = os.path.splitext(os.path.splitext(dataset_test_name)[0])[0]
    dataset_test, class_num, dataset_test_size = load_dataset(target_dir,dataset_test_folder, shuffle= True)
    print('Test Dataset Size:', len(dataset_test['target']))
    fig = plt.figure()
    plt.interactive(False)
    plt.imshow(dataset_test['input'][0])
    plt.title('Label: %d'%dataset_test['target'][0])
    plt.show()


if do_train == 1:
    dataset_train_folder = os.path.splitext(os.path.splitext(dataset_train_name)[0])[0]
    dataset_train,class_num, dataset_train_size = load_dataset(target_dir,dataset_train_folder)
    print('Train Dataset Size:', len(dataset_train['target']))
if do_userTest == 1:
    class_num = 10
    user_data_test, user_data_name, user_data_size = user_data_loader(user_test_dir)
    print('User Data Size:', user_data_size)
    if user_data_size == 0:
        raise ValueError('No data has been found in !', user_test_dir)
end = time.time()
print('Done within:', end-start)
# Data Preprocesses
#
def next_batch(dataset, batch_size, batch_begin):
    batch = {}
    batch_end = batch_begin+batch_size
    batch['input'] = dataset['input'][batch_begin:batch_end]
    target = np.array(dataset['target'][batch_begin:batch_end], dtype= 'int')
    batch['target'] = np.zeros((batch_size, class_num))
    batch['target'][np.arange(batch_size), target] = 1
    return batch

# Create Model
#
print('Create Model...')
start = time.time()
# I/O Place Holders
_input = tf.placeholder(tf.float32, [None, image_size, image_size])
_label = tf.placeholder(tf.float32, [None, class_num])
# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 200

with tf.device(device_type):
    weights = {
        'wc1':tf.Variable(tf.random_normal([3, 3, 1, 64], stddev= 0.1)),
        'wd1':tf.Variable(tf.random_normal([14*14*64, class_num], stddev= 0.1))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64], stddev= 0.1)),
        'bd1': tf.Variable(tf.random_normal([class_num], stddev= 0.1))
    }
    # Reshape
    _input_r = tf.reshape(_input, shape = [-1, image_size, image_size, 1])
    # Convolution
    _conv = tf.nn.conv2d(_input_r, weights['wc1'], strides = [1, 1, 1, 1], padding='SAME')
    # Adding Bias
    _bias = tf.nn.bias_add(_conv, biases['bc1'])
    # Relu
    _relu = tf.nn.relu(_bias)
    # Pooling
    _pool = tf.nn.max_pool(_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Vectorize
    _vector = tf.reshape(_pool, shape = [-1, weights['wd1'].get_shape().as_list()[0]])
    # Fully Connected
    _out = tf.add(tf.matmul(_vector, weights['wd1']), biases['bd1'])

    # Cost Function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= _label,logits= _out))
    # Optimizer
    optm = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
    # Count Correct
    _corr = tf.equal(tf.argmax(_out,1), tf.argmax(_label,1))
    # Correct Classified
    corr_classified = tf.reduce_mean(tf.cast(_corr, tf.float32))
    # Initializer
    init = tf.global_variables_initializer()
# Confusion Matrice
_conf = tf.confusion_matrix(tf.argmax(_label,1),tf.argmax(_out,1))
# Variable saver
saver = tf.train.Saver()
end = time.time()
print('Done within:', end-start)
#Train
#

with tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=NUM_THREADS)) as sess:
    sess.run(init)
    if do_train == 1:
        for epoch in range(training_epochs):
            avg_cost = 0
            corr_classified_percent = 0
            batch_begin = 0
            start = time.time()
            for itr in range(int(dataset_train_size/batch_size)):
                start_batch = time.time()
                batch = next_batch(dataset_train, batch_size, batch_begin)
                _,_cost,_corr_classified = sess.run([optm, cost, corr_classified], feed_dict={_input: batch['input'],
                                                                                             _label: batch['target']})
                avg_cost += _cost
                corr_classified_percent += _corr_classified
                batch_begin += batch_size
                end_batch = time.time()
               # print('batch:',itr, 'Cost:', _cost, ' Correct Classified Percentage:',
               #       _corr_classified, 'elapsed time:', end_batch - start_batch)
            corr_classified_percent = corr_classified_percent/(int(dataset_train_size/batch_size))
            avg_cost = avg_cost/dataset_train_size
            end = time.time()
            print('Train=> Epoch:', epoch+1, 'Average Cost:', avg_cost, ' Correct Classified Percentage:',
                  corr_classified_percent, 'elapsed time:', end-start)
            save_path = saver.save(sess, os.path.join(model_dir, model_name))

    if do_test ==1:
        saver.restore(sess, os.path.join(model_dir, model_name))
        avg_cost = 0
        corr_classified_percent = 0
        batch_begin = 0
        start = time.time()
        confusion_mat = np.zeros(shape=(class_num,class_num), dtype= int)
        for itr in range(int(dataset_test_size/batch_size)):
            batch = next_batch(dataset_test, batch_size, batch_begin)
            _cost, _corr_classified, _conf_mat = sess.run([cost, corr_classified, _conf], feed_dict={_input: batch['input'],
                                                                                   _label: batch['target']})
            avg_cost += _cost
            corr_classified_percent += _corr_classified
            batch_begin += batch_size
            confusion_mat += _conf_mat


        corr_classified_percent = corr_classified_percent / (int(dataset_test_size / batch_size))
        avg_cost = avg_cost / dataset_test_size
        end = time.time()
        print('Test=>', 'Average Cost:', avg_cost, ' Correct Classified Percentage:',
              corr_classified_percent, 'elapsed time:', end - start)
        print('Confusion Matrice:')
        print(confusion_mat)

    if do_userTest == 1:
        saver.restore(sess, os.path.join(model_dir, model_name))
        predict_label = sess.run([_out], feed_dict={_input: user_data_test}) #_label is not important
        for itr,item in enumerate(np.argmax(predict_label,axis = 2)[0,:]):
            print(user_data_name[itr], ':', chr(ord('A') + item))




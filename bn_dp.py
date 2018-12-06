#Eduardo F. Valdez
#MS Computer Science
#2012-97976
# CS 284 PA3 : CNN
#References can be found on the bottom of the file.


import os
import random
import tensorflow as tf
import cv2
import numpy as np
import time
import csv
from sklearn.model_selection import KFold
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib import slim


def getImagesLabels(data_dir, all=True):
    image_dir = os.path.join(data_dir, "Images")
    emotion_dir = os.path.join(data_dir, "Emotion")
    image = []
    label = []

    for root,dir,files in os.walk(emotion_dir):
        print(root)
        print(dir)
        print(files)
        if files:
            for file in files:
                if '.txt' in file: #if text file exists, as some sub folders do not have text file inside
                # example of a file name in emotions directory is S005/001/S005_001_00000011_emotion.txt with 3 as the label inside the .txt fle
                # Where the filename after the 2nd underscore _ ,00000011 means that image 1 to 11 is labelled 3
                dir = file.split("_")[0] + file.split("_")[1] #/S005/001
                count = file.split("_")[2]
                count = int(count)

                #image/S005/001
                emotionTextFile = os.path.join(emotionDirectory,dir)

                with open(emotionTextFile, 'r') as myfile:
                    data=myfile.read().replace('\n', '')
                    data=int(float(data))
                    print('value: ',data)
                print("Emotion File :",emotionTextFile)
                currentImageDirectory= os.path.join(imageDirectory,dir)
                print("Image Directory: ",currentImageDirectory)
                for root,dir,files in os.walk(currentImageDirectory)
                #get corresponding folder
                for iter in count:
                    pass
    #Use os.walk to go through all sub directories under Emotion Directory
    #https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    #root prints out directories only from what you specified
    #dirs prints out sub-directories from root
    #files prints out all files from root and directories
    #https://www.pythonforbeginners.com/code-snippets-source-code/python-os-walk/
    for root, dirs, files in os.walk(emotion_dir):
        #if len(files)> 0:
        if files:
            for file in files:

                basefile, ext = os.path.splitext(file)
                if ext == '.txt':
                    base = basefile.rsplit('_', 1)[0]
                    basedirs = base.split('_')
                    image_num = int(basedirs[2])

                    emotion_file = os.path.join(emotion_dir, basedirs[0], basedirs[1], file)

                    f = open(emotion_file, 'r')
                    for line in f:
                        line = line.strip()
                        line = int(float(line))
                        line = line - 1

                    if all == True:
                        temp_imagedir = os.path.join(image_dir, basedirs[0], basedirs[1])
                        for root, dirs, files in os.walk(temp_imagedir):
                            for file in files:
                                bfile, ext = os.path.splitext(file)
                                if ext == '.png':
                                    image.append(os.path.join(temp_imagedir, file))
                                    label.append(line)
                    else:
                        temp_imagedir_file = os.path.join(image_dir, basedirs[0], basedirs[1], "%s.png" % base)
                        image.append(temp_imagedir_file)
                        label.append(line)

    return image, label

def processImages(images):
 processedImages = []
 index = 0
 for image in images:
  image = cv2.imread(image, 0)
  image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
  processedImages.append(image)
  index += 1
  np.save("processed_images.npy", processedImages)
  return processedImages

def processLabels(labels):
 processedLabels = []
 for label in labels:
  one_hot = np.zeros(CLASSES)
  one_hot[label] = 1.0
  processedLabels.append(one_hot)
  np.save("processed_labels.npy", processedLabels)
  return processedLabels



def conv_net(x, filters, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    # Convolution 1
    #conv1 = conv2d(x, filters['1'], biases['1'], strides=2, padding=3, name="CONV1")
    W=tf.Variable(tf.truncated_normal([7,7,1,64], stddev=0.01), name="Filter1")
    b=tf.Variable(tf.truncated_normal([64], stddev=0.01), name="Bias1"),
    temp = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]])
    temp = tf.nn.conv2d(temp, W, strides=[1,2,2, 1], padding='VALID', name="Convolution1")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution1= tf.nn.relu(temp)

    # Poingoling 1
    #pool1 = maxpool2d(convolution1, k=3, strides=2, padding=1)
    convolution1 = tf.pad(convolution1, [[0,0],[1,1],[1,1],[0,0]])
    pooling1 = tf.nn.max_pool(convolution1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
    # LRN
    lrn = tf.nn.lrn(pooling1)
    # Convolution 2a
    #conv2a = conv2d(lrn, filters['2a'], biases['2a'], strides=1, padding=0, name="CONV2A")
    W=tf.Variable(tf.truncated_normal([1,1,64,96], stddev=0.01), name="Filter2a")
    b=tf.Variable(tf.truncated_normal([96], stddev=0.01), name="Bias2a")
    temp = tf.nn.conv2d(lrn, W, strides=[1,1,1,1], padding='VALID', name="Convolution2a")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution2a= tf.nn.relu(temp)

    # Pooling 2a
    pool2a = maxpool2d(lrn, k=3, strides=1, padding=1)
    temp = tf.pad(lrn, [[0,0],[1,1],[1,1],[0,0]])
    pooling2a = tf.nn.max_pool(temp, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')

    # Convolution 2b
    W=tf.Variable(tf.truncated_normal([3,3,96,208],stddev=0.01), name="Filter2b")
    b=tf.Variable(tf.truncated_normal([208],stddev=0.01), name="Bias2b")

    #conv2b = conv2d(conv2a, filters['2b'], biases['2b'], strides=1, padding=1, name="CONV2B")
    temp = tf.pad(convolution2a, [[0,0],[1,1],[1,1],[0,0]])
    temp = tf.nn.conv2d(temp, W, strides=[1,1,1, 1], padding='VALID', name="Convolution2b")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution2b= tf.nn.relu(temp)

    # Convolution 2c
    #conv2c = conv2d(pool2a, filters['2c'], biases['2c'], strides=1, padding=0, name="CONV2C")
    W=tf.Variable(tf.truncated_normal([1,1,64,64],stddev=0.01), name="Filter2c")
    b=tf.Variable(tf.truncated_normal([64],stddev=0.01), name="Bias2c")
    temp = tf.nn.conv2d(temp, W, strides=[1,1,1,1], padding='VALID', name="Convolution2c")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution2c= tf.nn.relu(temp)

    # Concatenate 2
    concat2 = tf.concat(3, [convolution2b, convolution2c])
    # Pooling 2b
    #pool2b = maxpool2d(concat2, k=3, strides=2, padding=1)
    temp = tf.pad(concat2, [[0,0],[1,1],[1,1],[0,0]])
    pooling2b = tf.nn.max_pool(temp, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # Convolution 3a
    #conv3a = conv2d(pooling2b, filters['3a'], biases['3a'], strides=1, padding=0, name="CONV3A")
    W=tf.Variable(tf.truncated_normal([1,1,64,64],stddev=0.01), name="Filter3a"),
    b=tf.Variable(tf.truncated_normal([64],stddev=0.01), name="Bias3a"),

    temp = tf.nn.conv2d(temp, W, strides=[1,1,1,1], padding='VALID', name="Convolution3a")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution3a= tf.nn.relu(temp)

    # Pooling 3a
    #pool3a = maxpool2d(pool2b, k=3, strides=1, padding=1)
    temp = tf.pad(pooling2b, [[0,0],[1,1],[1,1],[0,0]])
    pooling3a = tf.nn.max_pool(temp, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')


    # Convolution 3b
    #conv3b = conv2d(conv3a, filters['3b'], biases['3b'], strides=1, padding=1, name="CONV3B")
    W=tf.Variable(tf.truncated_normal([3,3,96,208],stddev=0.01), name="Filter3b")
    b=tf.Variable(tf.truncated_normal([208],stddev=0.01), name="Bias3b")
    temp = tf.pad(convolution3a, [[0,0],[1,1],[1,1],[0,0]])
    temp = tf.nn.conv2d(temp, W, strides=[1,1,1,1], padding='VALID', name="Convolution3a")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution3b= tf.nn.relu(temp)

    # Convolution 3c
    #conv3c = conv2d(pool3a, filters['3c'], biases['3c'], strides=1, padding=0, name="CONV3C")
    W=tf.Variable(tf.truncated_normal([1,1,272,64],stddev=0.01), name="Filter3c")
    b=tf.Variable(tf.truncated_normal([64],stddev=0.01), name="Bias3c")
    temp = tf.nn.conv2d(pooling3a, W, strides=[1,1,1, 1], padding='VALID', name="Convolution2b")
    temp = tf.nn.bias_add(temp, b)
    #APPLY BATCH NORMALIZATION
    temp = slim.batch_norm(temp)
    convolution3c= tf.nn.relu(temp)

    # Concatenate 3
    concat3 = tf.concat(3, [convolution3b, convolution3c])
    # Pooling 3b
    #pooling3b = maxpool2d(concat3, k=3, strides=2, padding=1)
    temp = tf.pad(concat3, [[0,0],[1,1],[1,1],[0,0]])
    pooling3b = tf.nn.max_pool(temp, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')

    # Fully Connected Layer
    pool_shape = pooling3b.get_shape().as_list()
    fc = tf.reshape(pooling3b, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
    fc = tf.add(tf.matmul(fc, tf.Variable(tf.truncated_normal([272*14*14, 11],stddev=0.01), name="FilterFullyConnected"), tf.Variable(tf.truncated_normal([11],stddev=0.01), name="BiasFullyConnected"))

    # Apply Dropout
    fc = tf.nn.dropout(fc, dropout)

    # Output/Prediction
    out = tf.add(tf.matmul(fc, tf.Variable(tf.truncated_normal([11, CLASSES],stddev=0.01), name="FilterOut"),tf.Variable(tf.truncated_normal([CLASSES],stddev=0.01), name="B_Out")))

    return out

# Parameters
LEARNING_RATE = 0.001
ITERATIONS=10
#ITERATIONS=30
#ITERATIONS = 50
#ITERATIONS=70
#ITERATIONS=100
BATCH_SIZE = 32
DISPLAY_STEP = 1
CLASSES = 7
CHANNELS = 1
IMAGE_SIZE = 224
N_INPUT = IMAGE_SIZE * IMAGE_SIZE
#DROPOUT=1 #NO DROPOUT
DROPOUT = 0.75


# Get File Paths and Labels
#Get current working director
currentDirectory = os.getcwd()
#Logging Directory is where we will save the logs such as copy of the model etc.
loggingDirectory = os.path.join(currentDirectory, "logdir")
images, labels = getImagesLabels(currentDirectory, True)
process_images(images)
process_labels(labels)
# Load processed data
processedImages = np.load("processed_images.npy")
processedLabels = np.load("processed_labels.npy")

#print("Saving processed data")
#trainimages, testimages, trainlabels, testlabels = do_kfold(processedImages, processedLabels, split=10)

if (os.path.exists('./trainimages.npy') and os.path.exists('./testimages.npy') and os.path.exists('./trainlabels.npy') and os.path.exists('./trainimages.npy')):
    print("Loading processed data. Will not pre-process data anymore.")
else:
    print("Will preprocess and save data.")
    #kf=KFold(n_splits=10,random_state=None,shuffle=False)
    kf=KFold(n_splits=10,random_state=None,shuffle=True)
    print(kf)
    for train_index, test_index in kf.split(processedImages):
        print("TRAIN:", train_index, "TEST:", test_index)
        trainingImages,testingImages = processedImages[train_index], processedImages[test_index]
        trainingLabels,testingLabels = processedLabels[train_index], processedLabels[test_index]
    kf.get_n_splits(processedImages)
    np.save("trainimages.npy", trainingImages)
    np.save("testimages.npy", testingImages)
    np.save("trainlabels.npy", trainingLabels)
    np.save("testlabels.npy", testingLabels)

#try to use kfold from sklearn

print("Loading processed data")
trainimages = np.load("trainimages.npy")
testimages = np.load("testimages.npy")
trainlabels = np.load("trainlabels.npy")
testlabels = np.load("testlabels.npy")

# tf Graph Input
x = tf.placeholder(tf.float32, [None, N_INPUT])
y = tf.placeholder(tf.float32, [None, CLASSES])
dropoutProb = tf.placeholder(tf.float32)

# Contruct Model
out = conv_net(x, filters, biases, dropoutProb)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
pred = tf.argmax(out, 1)
lab = tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init1 = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()

runtimes = []
# Select K Images and Labels
for k in range(1):
    start = time.clock()
    print("**************** kfold: %d ****************" % k)
    cost_periter = []
    acc_periter = []
    test_acc = []
    test_prediction = []
    test_label = []
    chkptfile = os.path.join(log_dir, "model%d_bn_dropout.ckpt" % k)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run([init1, init2])

        # Training Part
        for iter in range(ITERATIONS):
            tic = time.clock()
            train_len = len(trainimages[k])
            test_len = len(testimages[k])
            tmp_acc = []
            tmp_cost = []
            for num in range(int(train_len/BATCH_SIZE)):
                if num*BATCH_SIZE+BATCH_SIZE <= train_len:
                    end_batch = num*BATCH_SIZE+BATCH_SIZE
                else:
                    end_batch = train_len
                image_batch = trainimages[k][num*BATCH_SIZE:end_batch]
                label_batch = trainlabels[k][num*BATCH_SIZE:end_batch]

                x1 = np.array(image_batch).reshape(-1, N_INPUT)
                y1 = np.array(label_batch).reshape(-1, CLASSES)

                loss, opt, acc = sess.run([cost, optimizer, accuracy], feed_dict={x: x1, y: y1, dropoutProb: DROPOUT})
                cost_periter.append(loss)
                acc_periter.append(acc)
                tmp_acc.append(acc)
                tmp_cost.append(loss)

            avg_acc = sum(tmp_acc)/len(tmp_acc)
            avg_cost = sum(tmp_cost)/len(tmp_cost)
            toc = time.clock()
            if iter % DISPLAY_STEP == 0:
                print(iter, " accuracy: %.5f" % avg_acc, "cost: %.5f" % avg_cost, "time: %.1f s" % (toc-tic))
        # Testing Part
        for num in range(test_len):
            x2 = np.array(testimages[k][num]).reshape(-1, N_INPUT)
            y2 = np.array(testlabels[k][num]).reshape(-1, CLASSES)

            acc_test, prediction, labl = sess.run([accuracy, pred, lab], feed_dict = {x: x2, y: y2, dropoutProb: 1.})
            # print(k, num, "test accuracy: %.5f" % acc_test, "prediction: %.5f" % prediction, "label: %.5f" % labl)
            test_acc.append(acc_test)
            test_prediction.append(prediction[0])
            test_label.append(labl[0])

        # sess.close()
        saver.save(sess, chkptfile)

    outlist = zip(cost_periter, acc_periter)
    filename = "train-%d_bn_dropout.csv" % k
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(outlist)

    outtestlist = zip(test_acc, test_prediction, test_label)
    filename = "test-%d_bn_dropout.csv" % k
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(outtestlist)
    end = time.clock()
    runtime = end - start
    runtimes.append(runtime)
    print("time: %.2f s" % runtime)

with open("runtimes_bn_dropout.csv") as f:
    writer = csv.writer(f)
    writer.writerow(runtimes)



#References
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#https://stackoverflow.com/questions/36063014/what-does-kfold-in-python-exactly-do

from __future__ import print_function

import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.svm import SVC

def _extract_batches(images, ks):
    (num_img, height, width, channel) = images.shape

    idx = range(0, height - ks + 1, ks) # in case index of overflow
    idy = range(0, width - ks + 1, ks)
    id_iter = [(x, y) for x in idx for y in idy]

    batches = np.array([images[:,i:i+ks,j:j+ks,:] for (i, j) in id_iter])
    print("Extracted image batch shape: " + str(batches.shape))

    batches = np.reshape(batches, [-1, ks*ks*channel])
    print("Processed image batch shape: " + str(batches.shape))

    return batches

def _fit_anchor_vectors(batches, ks, channel_in, lossy_rate=1, augment=True):
    # remove mean
    # print("Image mean: " + str(np.mean(batches, axis=0)))
    batches = batches - np.mean(batches, axis=0)
    # print("Image mean after removal: " + str(np.mean(batches, axis=0)))

    # fit anchor vectors
    pca = PCA()
    pca.fit(batches)

    # get number of anchor vectors to keep based on lossy rate
    score = pca.explained_variance_ratio_
    components_to_keep = np.searchsorted(score, lossy_rate)
    print("Number of anchors to keep: " + str(components_to_keep))

    # get anchor vectors
    components = pca.components_[:components_to_keep,:]
    print("Anchor vector shape: " + str(components.shape))
    assert ks * ks * channel_in == components.shape[1]

    if augment:
        components = np.concatenate([components, -components], axis=0)
        components_to_keep = components_to_keep * 2
        print("Augmented anchor vector shape: " + str(components.shape))

    # reshape anchor vectors
    components = np.reshape(components, [-1, ks, ks, channel_in])
    print("Reshaped anchor shape: " + str(components.shape))

    # transpose anchor vectors to tensorflow format 
    # [ks, ks, channel_in, channel_out]
    components = components.transpose(1, 2, 3, 0)
    print("Tensorflow formated anchor shape: " + str(components.shape))

    # augment anchors
    return components, components_to_keep

def conv_and_relu(images, anchors, sess, ks):
    kernel = tf.constant(anchors)
    out = tf.nn.conv2d(images, kernel, strides=[1, 2, 2, 1], padding='SAME')
    out = tf.nn.relu(out)
    result = sess.run(out)
    print("Saak coefficients shape: " + str(result.shape))
    return result


def get_saak_anchors(images, _sess=None, ks=2):
    if _sess is None:
        sess = tf.Session()
    else:
        sess = _sess
    anchors = []
    channel_in = images.shape[3]

    rf_size = []
    n = min(images.shape[1], images.shape[2])
    while n >= ks:
        n = n // ks
        rf_size.append(n)

    print("Start to extract Saak anchors:\n")

    for i, _ in enumerate(rf_size):
        print("Stage %d start:" % (i + 1, ))
        batches = _extract_batches(images, ks)
        anchor, channel_out = _fit_anchor_vectors(batches, ks, channel_in)
        anchors.append(anchor)
        images = conv_and_relu(images, anchor, sess, ks)
        channel_in = channel_out
        print("Stage %d end\n" % (i + 1, )) 

    if _sess is None:
        sess.close()

    return anchors

def classify_svm(train_feature, train_label, test_feature, test_label):
    assert train_feature.shape[1] == test_feature.shape[1]
    assert train_feature.shape[0] == train_label.shape[0]
    assert test_feature.shape[0] == test_label.shape[0]
    svc = SVC()
    svc.fit(train_feature, train_label)
    accuracy = svc.score(test_feature, test_label)
    return accuracy

    


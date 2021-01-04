#%cd /face_detection/datasets/widerface/tf_records

import tensorflow as tf
import cv2

train_record = 'train.tfrecord'

def decode_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def read_tfrecord(example):
    features = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64)
            }
    tf_record = tf.io.parse_single_example(example, features)
    image = decode_image(tf_record['image/encoded'])

    height = tf.cast(tf.squeeze(tf_record['image/height']), tf.float32)
    width = tf.cast(tf_record['image/width'], tf.float32)
    xmin = tf.cast(tf_record['image/object/bbox/xmin'], tf.float32)
    xmax = tf.cast(tf_record['image/object/bbox/xmax'], tf.float32)
    ymin = tf.cast(tf_record['image/object/bbox/ymin'], tf.float32)
    ymax = tf.cast(tf_record['image/object/bbox/ymax'], tf.float32)
    
    labels = tf.stack([tf.sparse.to_dense(tf_record['image/object/bbox/xmin']),
             tf.sparse.to_dense(tf_record['image/object/bbox/ymin']),
             tf.sparse.to_dense(tf_record['image/object/bbox/xmax']),
             tf.sparse.to_dense(tf_record['image/object/bbox/ymax'])], axis=1)
    


    return image, labels # xmin,xmax,ymin,ymax

def load_dataset(filenames, ordered = False):    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = None) 
    dataset = dataset.with_options(ignore_order) 
    dataset = dataset.map(read_tfrecord, num_parallel_calls = None)
    dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.map(augment, num_parallel_calls = AUTO)
    
    
    # dataset = dataset.batch(10, drop_remainder=True)
    # dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

    

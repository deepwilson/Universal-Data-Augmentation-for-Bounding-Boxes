import tensorflow as tf
import cv2

train_record = 'train.tfrecord'

def decode_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.uint8)
    # img = tf.image.resize(img, (640,640))
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
    

    # image, labels = augment(image, labels)
    # image, image = transform(image, labels)
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

    
    
# BBox updating logic (universal)
def transform_(map, label, idx):
    shape = tf.shape(map)
    width, height = tf.cast(shape[1], dtype=tf.float32), tf.cast(shape[0], dtype=tf.float32)
    # tf.print('!!!!!', width, height)
    ''' Convert co-ordinates to absolute values '''
    xmin = label[0]*width
    ymin = label[1]*height
    xmax = label[2]*width
    ymax = label[3]*height
    xmin,xmax,ymin,ymax = tf.cast(xmin, tf.int32), tf.cast(xmax, tf.int32) , tf.cast(ymin, tf.int32) , tf.cast(ymax, tf.int32)

    ''' Get the indices of area b/w bbox co-ordinates '''
    indices = tf.meshgrid(tf.range(ymin, ymax), tf.range(xmin, xmax) , indexing='ij')
    indices = tf.stack(indices, axis=-1)

    ''' Create the mask '''
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    mask = tf.fill([bbox_h, bbox_w], idx)

    ''' Wear the mask '''
    masked_img = tf.scatter_nd(indices, mask, tf.shape(map))
    
    # print(idx,masked_img.shape, indices.shape, mask.shape,  shape)

    return masked_img, label, idx # [masked_img, label]


def retrieve_coords(map, idx, cds):
    pixel_value = idx # tf.constant(idx, dtype=tf.int32)
    cond = tf.where(tf.equal(map, pixel_value)) #[idx,:,:]
    # tf.print(cond)
    min  = tf.reduce_min(cond, axis=0)
    
    ymin, xmin = min[0], min[1]

    max = tf.reduce_max(cond, axis=0)
    ymax, xmax = max[0], max[1]
    # tf.print("----------------------------->min max", min, max)



    cds = tf.convert_to_tensor([xmin, xmax, ymin, ymax], dtype=tf.int32)
    # tf.print('$$$$$$', cds)
    return map, idx, cds


def augment(image, labels):
    # tf.print("image, labels before augment --------->", image, labels)
    # labels = tf.cast(labels, dtype=tf.float32)
    shape = tf.shape(image)[:-1]
    # tf.print('11111!!!!!', width, height)
    
    is_empty = tf.equal(tf.shape(labels)[0], 0)
    # print("labels --------->", labels)
    if is_empty:
        labels = tf.zeros([1,4], dtype=tf.float32)
    # tf.print("&shapeshape", tf.shape(labels), is_empty)
    label_img = tf.zeros([tf.shape(labels)[0], shape[0], shape[1]], dtype=tf.int32)
    
    range = tf.range(1, tf.shape(labels)[0]+1)

    label_img = tf.map_fn(lambda x:transform_(x[0],x[1], x[2]), (label_img, labels, range))[0]

    ''' Augment image+label_img '''
    label_img = tf.image.resize(tf.transpose(label_img, [1,2,0]), [640,640])
    image = tf.image.resize(image, [640,640])

    # tf.print('1] ******', image.shape, label_img.shape)
    # return image, labels
    image, label_img = transform(image, label_img)
    # image, label_img = transform(image, labels), transform(label_img, labels)


    label_img = tf.transpose(label_img, [2, 0, 1])
    label_img = tf.cast(label_img, dtype=tf.int32)
    # tf.print('******', image.shape, label_img.shape)

    coords = tf.zeros([tf.shape(labels)[0], 4], dtype=tf.int32)

    labels = tf.map_fn(lambda x:retrieve_coords(x[0], x[1], x[2]), (label_img, range, coords))[-1]
    # tf.print("^^^^^", labels)
    return image, labels

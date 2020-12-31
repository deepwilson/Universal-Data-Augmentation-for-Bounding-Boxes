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

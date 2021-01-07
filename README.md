## Need for ease:

---

Working on Object Detection pipelines, one of the main aspects that need to be dealt with is Data Augmentation. As compared to image classification, any augmentation applied to images in your object detection dataset should also be mapped with their respective bounding boxes.

The preferred way would be to write code that maps the destination bboxes for separate augmentations. For example, an augmentation which horizontally flips the image would require to bbox coordinates also to be flipped. We could write code for the same. However, every time we add an augmentation we need to write additional code which could be time-consuming.

This repo introduces a universal scheme for mapping the bbox coordinates as per the augmentation. 

## Logic:

---

The logic is simple:

- Consider a face detection setup
- Every face in the image would have a bbox
- For the 'n' faces in the image we would have 'n' bboxes
- Each bbox is plotted separately on an new image channel. Let's call this 'label_img'
- The 'label_img' will have 'n' channels (which is same as the number of bboxes) {This step ensured to take care of overlapping bboxes}
- The augmentations are applied to the 'label_img' and the bbox coordinates from the resulting 'label_img' after augmentation are recovered.

***The code has been written using TensorFlow ops so that it could be seamlessly integrated with tf.data pipeline.***

The results would look like this:

![1.png](1.png)

![2.png](2.png)

![3.png](3.png)

![4.png](4.png)

## How to use:
```
''' Augment image+label_img 
Whatever augmentation you want to apply to your image, apply them to label_imag as well
The retrieve_coords() function will take care of the mapping
For example I want to resize the image to (640,640) so I apply the same resizing function to the label_img as well '''

label_img = tf.image.resize(tf.transpose(label_img, [1,2,0]), [640,640])
image = tf.image.resize(image, [640,640])
```

```
''' Here we recover the bounding boxes from the label_img
The labels returned are the shifted/altered bbox coordinates'''
labels = tf.map_fn(lambda x:retrieve_coords(x[0], x[1], x[2]), (label_img, range, coords))[-1]
```

Whatever spatial augmentation you want to apply {Resizing, Flip left/right, Scaling, Translating, Rotation, Shearing, etc.} goes in to the "transform()" function. Make sure it takes 2 inputs (image, label_img)


## Dataset:

---

Dataset used is  -> [WiderFace](http://shuoyang1213.me/WIDERFACE/)

## 📒 Please refer the Colab notebook for the full pipeline and demo.

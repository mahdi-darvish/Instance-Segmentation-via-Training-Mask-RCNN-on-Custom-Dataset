# Instance Segmentation via Training Mask RCNN on Custom Dataset

 In this project, I tried to train a state-of-the-art convolutional neural network that was published in 2019. This model is well suited for instance and semantic segmentation. There is an option to use pre-trained weights. However, I took a step further and trained my own model using one of 600 classes from the Google Open Images dataset. I chose pumpkins as segmentation object, because why not? ;)
 
 
![thumbnail](/images/thumb.jpg)


## Requirements

Python 3.8.6, TensorFlow 2.4.1, Keras 2.4.3, and other common packages listed in requirements.txt.Using `tensorflow-gpu` is also an option that decreases the training time substantially.

You can use $ pip install -r requirements.txt inside your virtual environment to install them all or do it manually.

## Open Images Dataset V6 (2020)

Open Image is a humongous dataset containing more than 9 million images with respective annotations, and it consists of roughly 600 classes. I chose the pumpkin class and only downloaded those images, about 1000 images with the semantic and instance annotations. The dataset page itself won't support downloading by class category, and you need to use third-party applications; I used [OIDv6](https://pypi.org/project/oidv6/), which is a python libary that lets you download part of the dataset that you need, you may read more about that in Pypi page.

Another challenge was finding the masks. A CSV file with the data containing image IDs and respective mask annotations is not compatible with Mask R-CNN inputs. It would be best if you manipulated those. However, I used an alternative way: Downloading the mask images and finding all of the masks from a class for the specific embodiment, and assigning them to the `load_mask` function. You can find the code and explanation in the notebook itself.

## Introduction to Image Segmentation

In segmentation, we try to differentiate objects not just by creating a bounding box around them but rather by making pixel-wise masks around the object to show where exactly the object is located. To perform this task, we use convolutional neural network models, e.g., ResNet and AlexNet. After the fully connected layers, instead of predicting and classifying the image, we can upsample it and create the segmentation masks.

![segment](/images/segment.png)

### Instance vs Semantic Segmentation 
To be more precise, there are two different types of segmentation. Semantic segmentation focuses on creating a mask for all objects that fit in the same class and can not differentiate the instances of that object. However, Instance segmentation focuses on the countable objects and makes individual masks for each thing. 

![ivss](ivss.webp)

There are two technics to create the instance segmentation masks: Proposal-based and FCN-based.

The FCN-based method uses the fact that we can already create the semantic map of an image and create the instances by clustering the image and finding each object.
On the other hand, the proposal-based method takes advantage of bounding boxes around the object. Combining it with the semantic model, we can try to create a mask only within the ROI(region of interest) and create individual masks for each object.

You can find some papers about these methods also with better explanations than mine in the References section. All-in-all, both of these methods have their pros and cons. To take advantage of both of these methods simultaneously, the famous model Mask-RCNN was created.


## Mask R-CNN


## Results

Here, you can see the model's results on some test images.

![up-1](/images/up-1.jpg)

![up-2](/images/up-2.jpg)

![up-3](/images/up-3.jpg)


## Refrences

   ### Papers
   
> - [Mask R-CNN, 2019.](https://arxiv.org/pdf/1703.06870v3.pdf)
> - [Proposal-free Network for Instance-level Object Segmentation, 2015.](https://arxiv.org/pdf/1509.02636a)
> - [Simultaneous Detection and Segmentation, 2014.](https://arxiv.org/pdf/1407.1808)

  ### other refrences

> - [Mask R-CNN Repo](https://github.com/matterport/Mask_RCNN)
> - [Using Mask R-CNN with a Custom COCO-like Dataset](https://www.immersivelimit.com/tutorials/using-mask-r-cnn-on-custom-coco-like-dataset)
> - [Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)


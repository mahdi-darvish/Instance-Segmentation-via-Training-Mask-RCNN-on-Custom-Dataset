# Instance Segmentation via Training Mask RCNN on Custom Dataset

 In this project, I tried to train a state-of-the-art convolutional neural network that was published in 2019. This model is well suited for instance and semantic segmentation. There is an option to use pre-trained weights. However, I took a step further and trained my own model using one of 600 classes from the Google Open Images dataset. I chose pumpkins as segmentation object, because why not? ;)
 
 
![thumbnail](/images/thumb.jpg)


## Requirements

Python 3.8.6, TensorFlow 2.4.1, Keras 2.4.3, and other common packages listed in requirements.txt.Using `tensorflow-gpu` is also an option that decreases the training time substantially.

You can use $ pip install -r requirements.txt inside your virtual environment to install them all or do it manually.

## Open Images Dataset V6 (2020)

Open Image is a humongous dataset containing more than 9 million images with respective annotations, and it consists of roughly 600 classes. I chose the pumpkin class and only downloaded those images, about 1000 images with the semantic and instance annotations. The dataset page itself won't support downloading by class category, and you need to use third-party applications; I used [OIDv6](https://pypi.org/project/oidv6/), which is a python libary that lets you download part of the dataset that you need, you may read more about that in Pypi page.

Another challenge was finding the masks. A CSV file with the data containing image IDs and respective mask annotations is not compatible with Mask R-CNN inputs. It would be best if you manipulated those. However, I used an alternative way: Downloading the mask images and finding all of the masks from a class for the specific embodiment, and assigning them to the `load_mask` function. You can find the code and explanation in the notebook itself.


## Mask R-CNN


## Results

Here, you can see the model's results on some test images.

![up-1](/images/up-1.jpg)

![up-2](/images/up-2.jpg)

![up-3](/images/up-3.jpg)


## Refrences
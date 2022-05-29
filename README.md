# MScA_Deep_Learning_Project

## Inspiration

The idea is that when the government or city is doing a new urban planning it is important for them to know the pattern and distribution of zones in existing cities or areas. We think by using the neural network, we can identify the zoning of satellite pictures and try a mapping to see the zoning patterns of cities.

## Data Description 

3666 small satellite images of the whole city of Austin and a csv file which relates the name of each of these images to the corresponding zoning tag.
https://www.kaggle.com/datasets/franchenstein/austin-zoning-satellite-images

## Files

### zoning_image_processing_bounding_box_crop.ipynb

Since there is a good amount of overlap when it comes to the four different classes, we tried a bounding box crop in order to better allow the model to understand the main characteristics of the images. This also can help conserve resources when it comes to running and training the model. Ideally, by applying bounding boxes, the noise and unnecessary parts of the images would be discarded and the more important aspects would be highlighted.  


### zoning_satellite.ipynb

It is the main code for CNN model using Keras. The code includes the preprocessing for the label in the beginning. The model has 29,607,020 total params. The model ends up with relatively low accuracy score for training and validation; the loss is stable on training while it is fluctuating a lot for validation.

### pytorch_model.py

While the group has experience with Tensorflow, we wanted to experiment with another deep learning framework so we implemented a CNN in Pytorch. Pytorch has the advantage that it is built at a lower level so it gives the user more flexibility, at the cost of taking a bit more effort to implement. The model we used in that file is just another CNN architecture we tried, but it was helpful because we had problems configuring Tensorflow to use a GPU, whereas the Pytorch configuration was fairly easy. This allowed us to train on our local machines very quickly on large models that would otherwise have taken a while.

### vision_transformer.ipynb

Model with VisionTransformer. Consists of function for divide the image into patches and actual model. The model architecture is from the vision transformer from google research github page: https://github.com/google-research/vision_transformer. 

## Team

| Member                                                               
| :-------------------------------------------------------: 
              
| [Hyunjae Cho](https://github.com/hyunjaec)                
| [Milan Toolsidas](https://github.com/mtoolsidas)          
| [Dylaan Cornish](https://github.com/dylaancornish)   

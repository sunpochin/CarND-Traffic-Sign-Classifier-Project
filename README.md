# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[vis-image1]: ./pictures/traing_data.png "training data"
[vis-image2]: ./pictures/training_distribution.png "training data distribution"
[vis-image3]: ./pictures/valid_data.png "validation data bar chart"
[vis-image4]: ./pictures/valid_distribution.png "validation data distribution"
[vis-image5]: ./pictures/testing_data.png "testing data bar chart"
[vis-image6]: ./pictures/testing_distribution.png "testing data distribution"

[orig-image1]: ./pictures/orig-30-kmh-1.png "orig-30-kmh-1.png"
[gray-image1]: ./pictures/gray-30-kmh-1.png "gray-30-kmh-1.png"

[german-1]: ./pictures/german-traffic-signs/jpg/30.jpg "30.jpg"
[german-2]: ./pictures/german-traffic-signs/jpg/bumpy-road-1.jpg "bumpy-road-1.jpg"
[german-3]: ./pictures/german-traffic-signs/jpg/bumpy-road-2.jpg "bumpy-road-2.jpg"
[german-4]: ./pictures/german-traffic-signs/jpg/bumpy-road-3.jpg "bumpy-road-3.jpg"
[german-5]: ./pictures/german-traffic-signs/jpg/General-caution-1.jpg "Generalcaution-1"
[german-6]: ./pictures/german-traffic-signs/jpg/no-entry-1.jpg "no entry-1"
[german-7]: ./pictures/german-traffic-signs/jpg/priority-road.jpg "priority road"
[german-8]: ./pictures/german-traffic-signs/jpg/Right-of-way-next-intersection-1.jpg "Right-of-way at the next intersection-1"
[german-9]: ./pictures/german-traffic-signs/jpg/Right-of-way-next-intersection-2.jpg "Right-of-way at the next intersection-2"
[german-10]: ./pictures/german-traffic-signs/jpg/stop-1.jpg "stop-1"
[german-11]: ./pictures/german-traffic-signs/jpg/stop-2.jpg "stop-2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

#### 1. This README.md file.
#### 2. The jupyter notebook to run my model: [here](https://github.com/sunpochin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) .

### Dataset Exploration

#### 1. Dataset Summary: (Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.)

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32.
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory Visualization: (Include an exploratory visualization of the dataset.)

Here is an exploratory visualization of the data set. It is showing that all ID has some datas, zero data count could affect accuracy.
And the distributions are all right skewed.

![alt text][vis-image1]
![alt text][vis-image2]
![alt text][vis-image3]
![alt text][vis-image4]
![alt text][vis-image5]
![alt text][vis-image6]


### Design and Test a Model Architecture

#### 1. Preprocessing:

As first I tried to train the original LeNet-5 model with original images and the accuracy is about 70% which is way below the 93% threshold. So I read the forum to learn how to do preprocessing of grayscale and normalization.  

I normalized the image data because I tried to compare with the trained model validation accuracy and normalization really helped.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][orig-image1]
![alt text][gray-image1]


#### 2. Model Architecture: 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray-scale normalized image   							| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x64 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 11x11x128  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128  				|
| Fully connected		| outputs 3200       							|
| RELU					|												|
| Dropout				| 50% 											|
| Fully connected		| outputs 200       							|
| RELU					|												|
| Dropout				| 50% 											|
| Fully connected		| outputs 43       							|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet-5 model from last lab as a starting point.
I tried with various filter size for the Convolution Layer, and find out with depth size 32 I have an increased accuray about 93%.
Then I increased the depth size to 64 to get 95% accuracy, and tried with Dropout to get 1% more accuracy.  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
Ans: I used the LeNet-5 architecture as the first architecture to test the effect of preprocessing with grayscale and normalization. I choose it simply because I don't know anything else.

* What were some problems with the initial architecture?
Ans: The problem is it's accuracy is too low, at 70%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Ans: (Trying to answer above 3 questions) I tried to add a convolution layer but it doesn't help. 
And I tried to adjust filter width, height, depth, and found out bigger filter depth seem to help. I imagine that with increased filter depth, more details of different image characteristics could be stores in the network. 
I couldn't really grasp the idea of dropouts but it was mentioned in the class video and forum, so I added it, and it really pushed accuracy up about more that 1%.


(skip this question)
If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I download 11 German traffic sign .jpg files:

1. ![alt text][german-1] 
2. ![alt text][german-2] 
3. ![alt text][german-3]
4. ![alt text][german-4]
5. ![alt text][german-5]
6. ![alt text][german-6]
7. ![alt text][german-7]
8. ![alt text][german-8]
9. ![alt text][german-9]
10. ![alt text][german-10]
11. ![alt text][german-11]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the first 5 results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)	| Road work      								| 
| Bumpy road    		| Bumpy road									|
| Bumpy road			| Wild animals crossing							|
| Bumpy road      		| Bumpy Road					 				|
| General caution		| General caution      							|


The model was able to correctly guess 8 of the 11 traffic signs, which gives an accuracy of 72%, which is lower than the accuray on the test.

The prediction of No.1, No.3, and No.11 is wrong probably because of too much noise in the background. I should try to crop these 3 images and predict again. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is nearly 100% sure that this is a No.25 Road work but it's wrong, it's a Speed limit (30km/h). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0		      		| 25	Road work		 						| 
| 0			    		| 31	Wild animals crossing					|
| 0						| 10	No passing for vehicles over 3.5 metric tons|
| 0			      		| 23	Slippery road			 				|
| 0						| 30	Beware of ice/snow						|


For the second image the model is nearly 100% sure it's a Bumpy road and it's right this time:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0		      		| 22	Bumpy road		 						| 
| 0			    		| 29	Bicycles crossing						|
| 0						| 31	Wild animals crossing					|
| 0			      		| 26	Traffic signals			 				|
| 0						| 0	Speed limit (20km/h)						|


For the 3rd image the model is 95% sure it's a "Wild animals crossing" but it's wrong again, it's still a "Bumpy road".


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.95		      		| 31	Wild animals crossing					| 
| 0.003		    		| 40	Roundabout mandatory					|
| 0.001					| 18	General caution							|
| 0.0002	      		| 37	Go straight or left						|
| 0.00001				| 17	No entry								|


For the 4th image the model got it right at 99%, it's a "Bumpy road".

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.99		      		| 22	Bumpy road		 						| 
| 0			    		| 29	Bicycles crossing						|
| 0						| 31	Wild animals crossing					|
| 0			      		| 25	Road work		 						|
| 0						| 23	Slippery road							|


5th image the model predict it right at 99%.
Seems my model is so sure even when it's wrong, all the probability are above 95%.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.99		      		| 18	General caution							| 
| 0			    		| 27	Pedestrians								|
| 0						| 26	Traffic signals							|
| 0			      		| 37	Go straight or left		 				|
| 0						| 40	Roundabout mandatory					|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



# Ham-Spam-Message-Classifier
A machine learning model capable of predicting Hanm or Spam property of messages. The model is trained on two different supervised learning algorithm Support Vector Machine (SVM) and Logistic Regression (LR). 

The repository contain two different datasets files, a jupyter notebook containing all the code and two pickle files for SVC and LR  trained model

## Libraries 
### * Pandas
import pandas as pd
### * Matplotlib
import matplotlib as mpl

import matplotlib.pyplot as plt
### * Seaborn
import seaborn as sns
### * Numpy
import numpy as np
### * String
import string
### * NLTK
import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords
### * Wordcloud
from wordcloud import WordCloud
### * Sklearn
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
### * Pickle/joblib
import pickle


## The dataset
The Classifier is trained using two combined datasetwith two columns (a feature colmum "Message" and a label column "Category").


## Data Analysis
checking percentage of presence of LABELS (ham, spam) using Pie plot.

adding new column to the dataset 'length' with values of length of the text.

spam/ham message length histogram plot

checking number of spms and hams in the dataset using bar plot.


## Feather Engineering
to extract the tokens from the Message's column in every row, we use nltk method tokenize.word_tokenize().

reducing wods in each spam/ham messages to its word stem or simply reducing word to its base word.


## Data Visualization
to visualize text data we use worldcloud() technique.

checking most occuring words in spam messages and ham messages.

analyzing spam and ham messages with espect to he length of the text in each message using distribution plot.


## Model Training/spliting 
split the combined dataset into 20-80 ratio, the model will trained on SVM and Logistic Regression. Saved the trained model using pickle.
The two resulted model have accuracy on prediction 98.942 and 98.854 respectively.

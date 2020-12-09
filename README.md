# Ham-Spam-Message-Classifier
A machine learning model capable of predicting Han or Spam property of messages. The model is trained on two different supervised learning algorithm Support Vector Machine (SVM) and Logistic Regression (LR). 

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

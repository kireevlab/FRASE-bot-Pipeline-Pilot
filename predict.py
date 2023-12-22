import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, validation_curve

#print (tf.Session(config=tf.ConfigProto(log_device_placement=True)))
#opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

from keras.models import load_model

data=pd.read_table(sys.argv[1])
X_test=data.iloc[:,1:].to_numpy()
ids=data.iloc[:,0]

model=load_model('FRASE_model.h5')
y_pred=model.predict(X_test)

with open('FRASE_model_prediction.txt','w') as f:
	for i,pred in zip(ids,y_pred):
		f.write("{}\t{:.5f}\n".format(i,pred[0]))
f.close()


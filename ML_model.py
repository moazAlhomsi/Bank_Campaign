import streamlit as st
import joblib 
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report,plot_roc_curve
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

x_test= pd.read_csv('data/x_test.csv')
x_test = x_test.iloc[:,1:]
y_test= pd.read_csv('data/y_test.csv')
y_test = y_test.iloc[:,1:]
x_train= pd.read_csv('data/x_train.csv')
x_train = x_train.iloc[:,1:]
y_train= pd.read_csv('data/y_train.csv')
y_train = y_train.iloc[:,1:]

def run_ML_model():	
	st.header('ML Models Prediction')
	subb = st.sidebar.selectbox('Models',[
										  'Logistic Regression',
										  'Multi_layer Perceptron classifier',
										   ])

	if subb == 'Logistic Regression':
		with st.expander('Logistic Regression Model prediction'):
			model = load_model('model/LogisticRegression.pkl')

			pred=model.predict(x_train)
			pred_prob=model.predict_proba(x_train)
			score = roc_auc_score(y_train,pred_prob[:,1])
			st.markdown("The roc_auc score for the Train set is : {:.2%}".format(score))

			pred_prob=model.predict_proba(x_test)
			pred=model.predict(x_test)
			score = roc_auc_score(y_test,pred_prob[:,1])
			st.markdown("The roc_auc score for the Test set is : {:.2%}".format(score))

			st.write("_______________________________________________________________")
			st.write('Classification Report')
			st.text('______________________')
			st.text(classification_report(y_test,pred))
			st.write("_______________________________________________________________")			
			fig = plt.figure()
			plot_roc_curve(model,x_test,y_test)
			plt.title('ROC-AUC Curve')
			st.pyplot()


	if subb == 'Multi_layer Perceptron classifier':
		with st.expander('Multi_layer Perceptron classifier Model prediction'):
			model = load_model('model/MLPClassifier.pkl')

			pred=model.predict(x_train)
			pred_prob=model.predict_proba(x_train)
			score = roc_auc_score(y_train,pred_prob[:,1])
			st.markdown("The roc_auc score for the Train set is : {:.2%}".format(score))

			pred=model.predict(x_test)
			pred_prob=model.predict_proba(x_test)
			score = roc_auc_score(y_test,pred_prob[:,1])
			st.markdown("The roc_auc score for the Test set is : {:.2%}".format(score))

			st.write("_______________________________________________________________")
			st.write('Classification Report')
			st.text('______________________')
			st.text(classification_report(y_test,pred))
			st.write("_______________________________________________________________")			
			fig = plt.figure()
			plot_roc_curve(model,x_test,y_test)
			plt.title('ROC-AUC Curve')
			st.pyplot()

	# if subb == 'Random Forest':
	# 	with st.expander('Random Forest Clasifier Model prediction'):
	# 		model = load_model('model/RandomForest.pkl')

	# 		pred=model.predict(x_train)
	# 		pred_prob=model.predict_proba(x_train)
	# 		score = roc_auc_score(y_train,pred_prob[:,1])
	# 		st.markdown("The roc_auc score for the Train set is : {:.2%}".format(score))

	# 		pred=model.predict(x_test)
	# 		pred_prob=model.predict_proba(x_test)
	# 		score = roc_auc_score(y_test,pred_prob[:,1])
	# 		st.markdown("The roc_auc score for the Test set is : {:.2%}".format(score))

	# 		st.write("_______________________________________________________________")
	# 		st.write('Classification Report')
	# 		st.text('______________________')
	# 		st.text(classification_report(y_test,pred))
	# 		st.write("_______________________________________________________________")			
	# 		fig = plt.figure()
	# 		plot_roc_curve(model,x_test,y_test)
	# 		plt.title('ROC Curve')
	# 		st.pyplot()

	# 		imp = pd.Series(model.feature_importances_,
 #                index = x_test.columns)
	# 		imp.nlargest(8).plot(kind='barh')
	# 		plt.title('Feature Importance')
	# 		st.pyplot()		


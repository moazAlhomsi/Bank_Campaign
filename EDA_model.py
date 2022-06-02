import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set()
import scipy.stats as stats

df = pd.read_csv('data/Bank_Portugal.csv')
num_var = [var for var in df.columns if df[var].dtypes == "int64"]
cat_var = [var for var in df.columns if df[var].dtypes == "object"]
df_num = df[num_var]
df_cat=df[cat_var]
def run_EDA_model():

#	engin_types = ['Pre-engineering','Post-engineering']
	engin= st.sidebar.selectbox('Engineering of dataset',['Pre-engineering','Post-engineering'])
	if engin == 'Pre-engineering':
		col1,col2=st.columns(2)	
		with col1:
			with st.expander('Numerical variables'):
				st.write(num_var)
		with col2:
			with st.expander('Categorical variables'):
				st.write(cat_var)
		with st.expander('Description of Numerical data'):
			st.table(df[num_var].describe())


		# col3,col4=st.columns(2)
		# with col3:
		# 	with st.expander('Age boxplot'):
		# 		fig = plt.figure()
		# 		sns.boxplot(data = df_num['age'],orient='h',palette='ch:s=.25,rot=-.25')		
		# 		st.write(fig)

		# with col4:
		# 	with st.expander('Balance boxplot'):
		# 		fig = plt.figure()
		# 		sns.boxplot(data = df_num['balance'],orient='h',palette='ch:s=.25,rot=-.25')		
		# 		st.write(fig)	

		with st.expander('Subscription Ratio Pie Chart'):
			fig = plt.figure
			p1=df['subscribe'].value_counts().to_frame()
			p1 = p1.reset_index()
			p1.columns = ['Subscription','Counts']
			p01 = px.pie(p1,names='Subscription',values='Counts')
			st.plotly_chart(p01,use_container_width=True)	
		with st.expander("Diagnosing Outliers"):
			select = num_var
			option= st.radio("Choose a variable",select)

	#		if option:
			for i in select:
				if option == i:

				    # histogram
					    fig = plt.figure()
		#			    plt.subplot(1,3,1)
					    a=px.histogram(df,x=i,title='Histogram')
					    plt.title("Histogram")
					    st.plotly_chart(a,use_container_width=True)

	   
					    # Q-Q plot
					    fig = plt.figure(figsize=(5,2))
					    #plt.subplot(1,3,2)
					    b = stats.probplot(df[i],dist='norm',plot=plt)
					    plt.ylabel('Variable Quantiles')
					    st.pyplot(fig)

					    # Boxplot
					    fig = plt.figure()
					    plt.subplot(1,3,3)
					    c = px.box(df, y = i,title='Box Plot')
					    plt.title("Boxplot")
					    st.plotly_chart(c,use_container_width=True)



	if engin == 'Post-engineering':
		trimmed = pd.read_csv('data/TrimmedOutlierTrainSet.csv') 				    
		with st.expander("Diagnosing Outliers - Engineered"):
			select = num_var
			option= st.radio("Choose a variable",select)

	#		if option:
			for i in select:
				if option == i:

			    # histogram
				    fig = plt.figure()
	#			    plt.subplot(1,3,1)
				    a=px.histogram(trimmed,x=i,title='Histogram',nbins=50)
				    plt.title("Histogram")
				    st.plotly_chart(a,use_container_width=True)

   
				    # Q-Q plot
				    fig = plt.figure(figsize=(5,2))
				    #plt.subplot(1,3,2)
				    b = stats.probplot(trimmed[i],dist='norm',plot=plt)
				    plt.ylabel('Variable Quantiles')
				    st.pyplot(fig)

				    # Boxplot
				    fig = plt.figure()
				    plt.subplot(1,3,3)
				    c = px.box(trimmed, y = i,title='Box Plot')
				    plt.title("Boxplot")
				    st.plotly_chart(c,use_container_width=True)

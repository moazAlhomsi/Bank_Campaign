import streamlit as st
import streamlit.components.v1 as stc
st.set_page_config(page_title='Bank Campaign (portugal)',
					page_icon = '🏦')

from EDA_model import run_EDA_model
from ML_model import run_ML_model
html_temp = """
		<div style="background-color:#BE0000;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">ML predective model </h1>
		<h4 style="color:white;text-align:center;">Bank Marketing in Portugal </h4>
		</div>
		"""
menu = ['About','EDA','ML','Credits']
choice = st.sidebar.selectbox("Main",menu)
if choice == "About":
	# st.header('ML predective model')
	# st.subheader('Bank Marketing in Portugal')
	stc.html(html_temp)
	st.write("""

	The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
   in order to access if the product (bank term deposit) would be (or not) subscribed.
   	
 ### Input variables:
   #### bank client data:
   1 - age (numeric)

   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services")

   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

   4 - education (categorical: "unknown","secondary","primary","tertiary")

   5 - default: has credit in default? (binary: "yes","no")

   6 - balance: average yearly balance, in euros (numeric) 

   7 - housing: has housing loan? (binary: "yes","no")

   8 - loan: has personal loan? (binary: "yes","no")

   #### related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 

  10 - day: last contact day of the month (numeric)

  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

  12 - duration: last contact duration, in seconds (numeric)

  #### other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

  15 - previous: number of contacts performed before this campaign and for this client (numeric)

  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
	""")
	



if choice == 'EDA':
	run_EDA_model()

if choice == 'ML':
	run_ML_model()

if choice == 'Credits':
	st.subheader('Moaz Alhomsi - Head of Data Analytics ')
	st.text('___________________________________________')
	img='data/LOGO-AB-2-1.png'
	st.image(img)
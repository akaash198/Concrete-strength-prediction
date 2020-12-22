  #!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import base64
main_bg = "77.jpg"
main_bg_ext = "jpg"

side_bg = "77.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
components.html(
"""<h1 style="color:white;">CONCRETE STRENGTH PREDICTION</h1>
   """)


 
data=pd.read_excel("data.xlsx")  #reading dataset#
 

# user input fuction for getting input from user #
def user_input_features():
    Date=st.number_input("Enter the Date")
    Cement=st.number_input("Enter the Cement")
    Flyash=st.number_input("Enter the Flyash")
    Water=st.number_input("Enter the Water")
    CA=st.number_input("Enter the CA")
    FA=st.number_input("Enter the FA")
    
    data = {'Date':Date,
            'Cement':Cement,
            'Flyash':Flyash,
            'Water':Water,
            'CA':CA,
            'FA':FA
             }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
 

#machine learning process...#
X = data.iloc[:, 0:6].values
y = data.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X, y)  

 
y_pred = regressor.predict(df)
if st.button("PREDICT"): 
    st.subheader('PREDICTED VALUE')
    st.write(y_pred)#predicted output#
    st.subheader("PREDICTION ACCURACY")
    st.write(regressor.score(X,y)*100)#accuracy of this ML prediction#
 






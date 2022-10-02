import streamlit as st
import numpy as np
import pandas as pd
from pickle import load

df=pd.read_csv('Data/diamonds.csv')

# Loading pretrained models from pickle file
ord_enc=load(open('models/ordinal_encoder.pkl','rb'))
scaler = load(open('models/standard_scaler.pkl', 'rb'))
dt_regressor=load(open('models/dt_model.pkl','rb'))


st.title('💎 Diamond Price Prediction 💎')

'Please fill all the Diamond properties.'

with st.form('my_form'):
    carat = st.number_input('Enter Carat Value')
    cut = st.selectbox(label='Select Cut of Diamond', options=df.cut.unique())
    color = st.selectbox(label='Select Color of Diamond', options=df.color.unique())
    clarity= st.selectbox(label='Select Clarity level of Diamond', options=df.clarity.unique())
    depth=st.number_input('Enter depth value of Diamond')
    table=st.number_input('Enter table value of Diamond')

    
    btn = st.form_submit_button(label='Predict')


if btn:
    if carat and cut and color and clarity and depth and table:
        query_num = pd.DataFrame({'carat':[carat], 'depth':[depth],'table':[table]})
        query_cat = pd.DataFrame({'cut':[cut], 'color':[color], 'clarity':[clarity]})
        query_cat = ord_enc.transform(query_cat)
        query_num = scaler.transform(query_num)
        query_point = pd.concat([pd.DataFrame(query_num), pd.DataFrame(query_cat)], axis=1)
        price = dt_regressor.predict(query_point)
        
        st.success(f"The price of Selected Diamond is $ {round(price[0]-172,2)}",icon="✔️")
        st.balloons()
    else:
        st.error('Enter all the values')    
        st.snow()

   

################################## BACKGROUND IMAGE CODE #######################################

page_bg_img = '''
<style>

.stApp {
<font color="blue"> 
background-image: url("https://physics.aps.org/assets/228bead6-f070-4f7d-b18f-c876e70b8ac8/e40_1_medium.png");
background-size: cover;

}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
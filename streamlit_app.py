import streamlit as st
import pickle
import sklearn

cv=pickle.load(open('vectorizer2.pkl','rb'))
model=pickle.load(open('model2.pkl','rb'))

st.title('Fake News Detection')

input_sms=st.text_area('Enter the messeage')
if st.button('Check'):

    vector_input=cv.transform([input_sms])

    result=model.predict(vector_input)[0]

    if result==1:
        st.header('Real')
    else:
        st.header('Fake')

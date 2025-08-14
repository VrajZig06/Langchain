import streamlit as st
import requests
import json

st.title("Vraj's POSTMAN")
REQUEST_TYPE = st.selectbox("Request Type",['GET','POST','DELETE','PUT','PATCH'])
REQUEST_URL  = st.text_input("Enter your URL Here")

SEND_BUTTON = st.button("Send")

match REQUEST_TYPE:
        case "GET":
            if SEND_BUTTON:
                response = requests.get(REQUEST_URL)
                data = response.json()
                st.write(data)
        case "POST":
            json = st.text_area("Enter Paylaod")
            if SEND_BUTTON:
                response = requests.post(REQUEST_URL)
                st.write(f"Response : ")
                st.write(json)
        case "DELETE":
            response = requests.delete(REQUEST_URL)
            st.write(response.json())
        case "PUT":
            response = requests.put(REQUEST_URL)
            st.write(response.json())
        case "PATCH":
            response = requests.patch(REQUEST_URL)
            st.write(response.json())




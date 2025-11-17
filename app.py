import streamlit as st
import requests
 
API_URL = "http://127.0.0.1:8051/get-email"
 
st.set_page_config(layout="wide", page_title="Email Generator")
# st.markdown("""
# <div style='text-align: center;'>
#     <img src='https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg' width='50'/>
# <h2 style='color:skyblue;'>Profile-Based Customized Emails for Mass Mailing</h2>
# </div>
# """, unsafe_allow_html=True)
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; gap: 10px;'>
    <h2 style='color:skyblue; margin: 0;'>Profile-Based Customized Emails for Mass Mailing</h2>
    <img src='https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg' width='40'/>
</div>
""", unsafe_allow_html=True)
 
left, right = st.columns(2)
 
with left:
    st.header("Input Panel")
 
    email_id = st.text_input("Email ID")
    subject = st.text_input("Subject")
    body = st.text_area("Email Body", height=150)
  
    if st.button("Send"):
 
        payload = {
            "email_id": email_id,
            "subject": subject,
            "body": body
        }
        response = requests.post(API_URL, json=payload)

        if response.status_code == 250:
            st.session_state["generated"] = response.json()["result"]
        else:
            st.error("Error calling backend")



    Query = st.text_input("Query")
    # st.button("Ask")

    
    # st.button("Generate Summery of the day")
    # st.write()



with right:
    st.header("Output Panel")
    
    st.text_area(
        "Generated Output/Query Output", 
        value=st.session_state.get("generated", ""), 
        height=250
    )

    # st.text_area(
    #     "Query Output", 
    #     value=st.session_state.get("generated", ""), 
    #     height=350
    # )

    st.button("Generate Summery of the day")
    # st.write()
 
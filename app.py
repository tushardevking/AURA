# Streamlit — web UI library hai Python ki
# 'app' — hamara LangGraph agent hai agent.py se
import streamlit as st
import tempfile  # temporary file system pe save karne ke liye
from agent import app  # LangGraph compiled graph import kiya

# Page ka title aur subtitle
st.title("AURA")
st.caption("Your personal data analyst")

# File uploader — user CSV upload karega
# type=["csv"] — sirf CSV allowed hai
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Chat input — user question likhega
# ye Streamlit ka built-in chat input box hai
question = st.chat_input("Ask anything...")

# Sirf tab chalao jab dono ho — file bhi aur question bhi
# agar koi ek missing hai toh agent crash karega
if uploaded_file and question:
    
    # Streamlit file directly disk pe nahi hoti — memory mein hoti hai
    # LangGraph ko disk path chahiye — isliye temporarily save karo
    # 'with' block khatam hone pe automatically close hoga file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())  # file content disk pe likho
        tmp_path = tmp.name  # path save karo — agent ko denge

    # LangGraph agent invoke karo — poora pipeline chalega
    # Question aur filepath dono pass karo State mein
    # baaki fields empty — nodes khud bharenge
    with st.spinner("AURA is analyzing..."):  # loading indicator
        result = app.invoke({
            "Question": question,
            "filepath": tmp_path,  # user ki uploaded file ka path
            "data": "",            # dataloader bharega
            "code": "",            # code_writer bharega
            "analysis": "",        # code_executer bharega
            "insights": "",        # insights_writer bharega
            "chart": ""            # chartmaker bharega
        })

    # Insights dikhao — MBB style consulting text
    st.subheader("Insights")
    st.write(result["insights"])

    # Plotly chart dikhao — interactive chart
    st.subheader("Chart")
    st.plotly_chart(result["chart"])
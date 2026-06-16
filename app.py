import streamlit as st
import tempfile
from agent import app

st.title("AURA")
st.caption("Your personal data analyst")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Chat history — session mein store karo taaki reload pe na jaaye
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani conversation dikhao
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask anything...")

if uploaded_file and question:
    
    # User message dikhao aur save karo
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("AURA is analyzing..."):
            result = app.invoke({
                "Question": question,
                "filepath": tmp_path,
                "data": "",
                "code": "",
                "analysis": "",
                "insights": "",
                "chart": ""
            })

        # Assistant response dikhao
        with st.chat_message("assistant"):
            st.write(result["insights"])
            st.plotly_chart(result["chart"])
        
        # History mein save karo
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["insights"]
        })

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
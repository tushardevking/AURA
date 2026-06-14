from typing import TypedDict
import pandas as pd
import os
from dotenv import load_dotenv as de
from groq import Groq

de()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))
class aurastate (TypedDict):
    Question: str
    data: str
    code: str
    analysis: str
    insights: str

def dataloader(state: aurastate) :
    df=pd.read_csv("HR_Employee_Attrition.csv")
    col=df.columns.tolist()
    return {"data": f"Columns: {str(col)}, Rows: {len(df)}"}

def code_writer(state: aurastate):
    question=state["Question"]
    data=state["data"]
    message=[{"role": "system","content":"""You are a Python data analyst.
User will give you a question and dataset columns.
Return ONLY executable Python pandas code — no explanation, no markdown, no text.
We have already imported pandas as pd and the dataframe is already loaded as variable 'df'.
Do not include import pandas and pd.read_csv() in your code.Do not use print() statements. Store the final result in a variable called 'result'. Also it should only contain code no backticks anything only simple code""" }]
    message.append({"role": "user","content": question+ data})
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message
    )
    return {"code": response.choices[0].message.content}

def code_executer(state: aurastate):
    code=state["code"]
    df = pd.read_csv("HR_Employee_Attrition.csv")
    safe_env={
        "df": df,
        "pd": pd
    }
    exec(code, safe_env)
    result=safe_env["result"]
    return {"analysis": result}

def insights_writer(state: aurastate):
    question=state["Question"]
    code=state["code"]
    analysis=state["analysis"]
    message=[{"role":"system","content":"You're a management consultant and your client has asked you the question for that you have the question, the code and the analysis code generated now like MBB consultant rewrite the insights from the analysis into proper MMB style consulting insights which is easier for the client to understand"}]
    message.append({"role":"user", "content": question + code + str(analysis)})
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message
    )

    return{"insights": response.choices[0].message.content}

test_state = {
    "Question": "show attrition by department",
    "data": "",
    "code": "",
    "analysis": "",
    "insights": ""
}

state_after_load = dataloader(test_state)

# Step 2 — state merge karo
test_state.update(state_after_load)

# Step 3 — code_writer chalao updated state ke saath
state_after_code = code_writer(test_state)
test_state.update(state_after_code)
state_after_analysis=code_executer(test_state)
print("Generated code:")
print(test_state["code"])
test_state.update(state_after_analysis)
state_after_insight=insights_writer(test_state)
print(state_after_insight)
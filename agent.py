from typing import TypedDict, Any
import pandas as pd
import os
from dotenv import load_dotenv as de
from groq import Groq
from langgraph.graph import StateGraph, START, END
import plotly.express as px


de()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))
class aurastate (TypedDict):
    Question: str
    data: str
    code: str
    analysis: Any
    insights: str
    chart: Any

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

def chartmaker(state: aurastate):
    question=state["Question"]
    analysis=state["analysis"]
    insights=state["insights"]
    message=[{"role": "system", "content": """
                  You're an expert data analyst, user shares their question, data they gto from pandas and insights. your job is to turn that data into masterful aesthectically pleasing and simple to understand chart code of plotly.express. store the code in variable called fig, don't use any kind of backticks while returning the answer just simple code stored in fig variable"""}]
    message.append({"role": "user", "content": question+str(analysis)+insights})
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message
        )
    safe_env={
            "px":px,
            "pd":pd,
            "result":analysis,
            "df": pd.read_csv("HR_Employee_Attrition.csv")  # ye add karo

        }
    exec(response.choices[0].message.content, safe_env)
    chart=safe_env["fig"]
    return {"chart": chart}
        


ques=str(input("Please enter your question: "))
graph=StateGraph(aurastate)
graph.add_node("dataloader", dataloader)
graph.add_node("code_writer", code_writer)
graph.add_node("code_executer", code_executer)
graph.add_node("insights_writer", insights_writer)
graph.add_node("chartmaker", chartmaker)

graph.add_edge("dataloader","code_writer")
graph.add_edge("code_writer","code_executer")
graph.add_edge("code_executer","insights_writer")
graph.add_edge("insights_writer","chartmaker")

graph.add_edge(START, "dataloader")
graph.add_edge("chartmaker", END)

app=graph.compile()
result=app.invoke({
    "Question": ques,
    "data": "",
    "code": "",
    "analysis": "",
    "insights": "",
    "chart": ""
})

print(result["chart"]) 
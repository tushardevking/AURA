from typing import TypedDict, Any
import pandas as pd
import os
from dotenv import load_dotenv as de
from groq import Groq
from langgraph.graph import StateGraph, START, END
import plotly.express as px
import plotly


de()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))
class aurastate (TypedDict):
    Question: str
    filepath: str
    data: str
    code: str
    analysis: Any
    insights: str
    chart: Any

def dataloader(state: aurastate) :
    df=pd.read_csv(state["filepath"])
    col=df.columns.tolist()
    return {"data": f"Columns: {str(col)}, Rows: {len(df)}"}

def code_writer(state: aurastate):
    question=state["Question"]
    data=state["data"]
    message=[{"role": "system","content":"""You are a Python data analyst.
User will give you a question and dataset columns.
Return ONLY executable Python pandas code — no explanation, no markdown, no text.
We have already imported pandas as pd and the dataframe is already loaded as variable 'df'.
Do not include import pandas and pd.read_csv() in your code.Do not use print() statements. Store the final result in a variable called 'result'. Also it should only contain code no backticks anything only simple code.
If a column contains text values like 'Yes'/'No', convert to numeric first using: df['column'] = df['column'].map({'Yes': 1, 'No': 0})""" }]
    message.append({"role": "user","content": question+ data})
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message
    )
    return {"code": response.choices[0].message.content}

def code_executer(state: aurastate):
    code=state["code"]
    df = pd.read_csv(state["filepath"])
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
You are an expert data visualization specialist.

Your job is to write Plotly Express code to visualize pandas data.

WHAT YOU WILL RECEIVE:
- User's question
- Pandas analysis result
- Business insights text

STRICT RULES:
- Only use 'px' for all charts — it is already imported as plotly.express
- Never write 'import' anything — px, pd, df, result are all pre-loaded
- Never use plotly.express directly — always use px
- Never use print() statements
- Never use backticks or markdown
- Store final chart in variable called 'fig'
- Return only executable Python code — nothing else

CHART SELECTION RULES:
- Comparing categories → px.bar()
- Trend over time → px.line()
- Distribution → px.histogram()
- Part of whole → px.pie()
- Two numeric columns → px.scatter()

CHART QUALITY RULES:
- Always add a clear title
- Always label x and y axes
- Use color_discrete_sequence for better colors
- Keep it clean and professional
IMPORTANT: The 'result' variable may be an aggregated DataFrame. 
If you need original columns like JobRole, use 'df' instead of 'result'.
Always check what columns are available before plotting.
              Final Note: Only return the code no backticks no explanation just python executable code in fig variable so we don't get any runtime or any other kind of error              
              
"""}]
    message.append({"role": "user", "content": question+str(analysis)+insights})
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message
        )
    safe_env = {
    "px": px,
    "plotly": plotly,  # ye add karo
    "pd": pd,
    "result": analysis,
    "df": pd.read_csv(state["filepath"])
}
    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").strip()
    exec(code, safe_env)
    chart=safe_env["fig"]
    chart.show()  # ye add karo
    return {"chart": chart}
        
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

from dotenv import load_dotenv
import os
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import base64
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# Load credentials from .env
load_dotenv()

# ✅ Query helper
def query_snowflake(query: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def generate_chart(df, metric="MARKETCAP") -> dict:
    """
    Generates chart and returns both file path and base64 encoded image
    Returns: {
        "file_path": "path/to/chart.png", 
        "base64": "data:image/png;base64,...",
        "message": "Chart saved as path/to/chart.png"
    }
    """
    # Existing plotting code (unchanged)
    df = df.sort_values("ASOFDATE")
    df["ASOFDATE"] = pd.to_datetime(df["ASOFDATE"])
    
    plt.figure(figsize=(10, 4))
    plt.plot(df["ASOFDATE"], df[metric], marker="o", linewidth=2)
    plt.title(f"{metric} Over Time")
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B'))
    plt.tight_layout()
    
    # 1. Keep original file save functionality
    chart_path = f"{metric.lower()}_chart.png"
    plt.savefig(chart_path)
    
    # 2. New: Generate base64 without affecting original flow
    buf = plt.gcf().canvas.buffer_rgba()
    img_base64 = base64.b64encode(buf).decode('utf-8')
    
    plt.close()
    
    return {
        "file_path": chart_path,
        "base64": f"data:image/png;base64,{img_base64}",
        "message": f"Chart saved as {chart_path}"  # Maintains original return message
    }


# ✅ LangChain Tool
@tool
def get_nvidia_financials(input: str) -> str:
    """
    Get NVIDIA financials from Snowflake for a given year and quarter.
    Input format: "year=2024, quarter=1"
    """
    try:
        year = input.split("year=")[1].split(",")[0].strip()
        quarter = input.split("quarter=")[1].strip()

        query = f"""
        SELECT * FROM NVIDIA_FINANCIALS 
        WHERE YEAR(ASOFDATE) = {year} AND QUARTER(ASOFDATE) = {quarter}
        """

        df = query_snowflake(query)

        if df.empty:
            return f"No data found for year {year} and quarter {quarter}."

        # Text summary
        row = df.iloc[0]
        summary = (
            f"NVIDIA Financials for Q{quarter} {year}:\n"
            f"- ASOFDATE: {row['ASOFDATE']}\n"
            f"- ENTERPRISEVALUE: {row['ENTERPRISEVALUE']:,}\n"
            f"- MARKETCAP: {row['MARKETCAP']:,}\n"
            f"- PERATIO: {row['PERATIO']:.2f}\n"
            f"- PBRATIO: {row['PBRATIO']:.2f}\n"
            f"- PSRATIO: {row['PSRATIO']:.2f}\n"
            f"- PEGRATIO: {row['PEGRATIO']:.4f}\n"
            f"- FORWARDPERATIO: {row['FORWARDPERATIO']:.2f}"
        )

        # Save chart
        chart_msg = generate_chart(df, metric="MARKETCAP")

        return summary + f"\n\n📊 {chart_msg}"

    except Exception as e:
        return f"Error parsing input or querying Snowflake: {e}"

# 🔮 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 🤖 Agent
agent = initialize_agent(
    tools=[get_nvidia_financials],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def get_nvidia_financial_response(user_query: str) -> dict:
    """Process user query through the pre-initialized agent."""
    return agent.invoke({"input": user_query})

# Simple test
user_prompt = "Provide summary of 2024 with the quarters"
response = get_nvidia_financial_response(user_prompt)
print(response["output"])
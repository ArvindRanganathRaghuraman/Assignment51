from dotenv import load_dotenv
import os
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# ✅ Chart generator (saves to PNG)
def generate_chart(df, metric="MARKETCAP") -> str:
    df = df.sort_values("ASOFDATE")
    df["ASOFDATE"] = pd.to_datetime(df["ASOFDATE"])

    plt.figure(figsize=(10, 4))
    plt.plot(df["ASOFDATE"], df[metric], marker="o", linewidth=2)
    plt.title(f"{metric} Over Time")
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)

    # Format y-axis (e.g., 2.3B)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B'))

    plt.tight_layout()
    chart_path = f"{metric.lower()}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return f"Chart saved as {chart_path}"

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

# 🔁 Prompt the user via terminal
user_prompt = input("Your question: ")
response = agent.invoke(user_prompt)
print(response["output"])

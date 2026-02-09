from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient
from datetime import datetime
from dotenv import load_dotenv
import os


load_dotenv()

def print_message(respone):
    """to extract only the needed output from the llm respone"""
    messages = respone.get("messages", [])
    final_text = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and "text" in item:
                        final_text += item["text"]
            else:
                final_text = str(msg.content)
            break

    print(final_text)


# Initialize
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Tool 1: Web Search
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, and facts."""
    results = tavily_client.search(query, max_results=3)
    formatted = []
    for r in results['results']:
        formatted.append(f"Title: {r['title']}\nContent: {r['content']}\n")
    return "\n---\n".join(formatted)

# Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '365 - 100' or '50 * 2'"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 3: Current Date
@tool
def get_current_date(dummy: str = "") -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


@tool
def get_current_date_with_timezone(dummy: str = "", timezone:str="", country:str="") -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

# Create tools list
tools = [web_search, calculator, get_current_date, get_current_date_with_timezone]

# Create agent with Langchain (one line!)
agent_executor = create_agent(llm, tools)

# Test queries
if __name__ == "__main__":
    queries = [
        # "What is today's date?",
        # "How many days until December 31, 2025?",
        # "What are the latest AI developments this week?",
        # "Find the USD to INR currency exchange rate of only today and convert 100 usd to INR",
        "What is the price of SBI share today vs last week same day. provide data in tabular format."
    ]
    
    print("=" * 80)
    for i, query in enumerate(queries, 1):
        if not query:
            print("query missing....")
            continue
        print(f"\n🔍 QUERY {i}: {query}\n")
        
        # Stream the agent's response
        for chunk in agent_executor.stream(
            {"messages": [("user", query)]},
            stream_mode="values"
        ):
            if "messages" in chunk:
                chunk["messages"][-1].pretty_print()

        print("=" * 80)
        print("=" * 80)
        # print only final output - without streaming 
        result = agent_executor.invoke({"messages": [("user", query)]})
        print_message(result)
       

        print("=" * 80)
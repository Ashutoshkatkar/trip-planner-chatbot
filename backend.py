import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
# from tools.tavily_tool import tavily_search
from mcp_test_client import tavily_mcp_search
import asyncio

def get_databse_url():
    databse_url=os.getenv("DATABASE_URL")

    if not databse_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add it to render pgsql external database"
        )
    if "sslmode=" not in databse_url:
        seperator="&" if "?" in databse_url else "?"
        databse_url=f"{databse_url}{seperator}sslmode=require"
    return databse_url


GROQ_API_KEY=os.getenv("GROQ_API_KEY")


#LLM
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

#state
class Travelstate(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    hotel_results:str
    itinerary:str
    llm_calls:int


#hotel agent
def hotel_agent(state:Travelstate):
    query=f"best hotels in {state['user_query']}"
    # hotel_results=tavily_search(query)
    hotel_results=asyncio.run(tavily_mcp_search(query))

    return{
        "hotel_results":hotel_results,
        "messages":[
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls":state.get("llm_calls",0)+1
    }

#itinerary agent

def itinerary_agent(state:Travelstate):
    prompt=f""" 
    Create an complete travel itinerary

    User Query:
    {state['user_query']}

    Hotel Results:
    {state['hotel_results']}

    Make the itinerary practical,budget aware and easy to follow

    """
    response=llm.invoke([
        SystemMessage(content="You are an expert travel planner"),
        HumanMessage(content=prompt)
    ])
    return{
        "itinerary":response.content,
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }

#final agent

def final_agent(state: Travelstate):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Hotel Suggestions
3. Day-by-Day Itinerary
4. Estimated Budget
5. Final Recommendations

Important:
- Be clear and practical.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

#graph

graph=StateGraph(Travelstate)

graph.add_node("hotel_agent",hotel_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent",final_agent)

graph.add_edge(START,"hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)

#postgres checkpointer

DATABASE_URL=get_databse_url()

_conn=psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer=PostgresSaver(_conn)
checkpointer.setup()

travel_graph=graph.compile(checkpointer=checkpointer)


#function for fastapi

def run_travel_agent(user_input:str,thread_id:str):
    config={
        "configurable":{
            "thread_id":thread_id
        }
    }

    result=travel_graph.invoke(
        {
            "messages":[
                HumanMessage(content=user_input)
            ],
            "user_query":user_input,
            "hotel_results":"",
            "itinerary":"",
            "llm_calls":0
        },
        config=config
    )
    final_answer=result["messages"][-1].content

    return{
        "thread_id":thread_id,
        "answer":final_answer,
        "hotel_results":result.get("hotel_results",""),
        "itinerary":result.get("itinerary",""),
        "llm_calls":result.get("llm_calls",0),
    }
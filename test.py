from tools.tavily_tool import tavily_search
from backend import run_travel_agent


user_ip="travel to japan"
res=run_travel_agent(
    user_input=user_ip,
    thread_id="1"
)

print(res['answer'])



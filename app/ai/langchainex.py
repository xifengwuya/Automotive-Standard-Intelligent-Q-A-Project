import os
import openai

from dotenv import load_dotenv, find_dotenv
from langchain.agents.factory import create_agent
from langchain_core.prompt_values import ChatPromptValue
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

checkpointer = InMemorySaver()

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.getenv('API_KEY')
llm_model = os.getenv('LLM_MODEL')
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


model = ChatOpenAI(
    api_key=os.getenv('API-KEY'),
    base_url = os.getenv('API_URL'),
    temperature=0.0,
    model=llm_model,
    max_tokens=100,
    timeout=10
)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])






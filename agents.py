import os
import json
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import random
from sympy import sympify  # Safe math evaluation

# Load API Key Securely
os.environ['GOOGLE_API_KEY'] = st.secrets["AIzaSyBzovUvxbHB7sF09ivq_wLM-g6tB3bOOsg"]

# Initialize Model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")

###### AGENT-1 General Query Agent ######

def general(input=""):
    try:
        results = model.invoke(input)
        return results.content
    except Exception:
        return "Sorry, I couldn't understand your query."

general_tool = Tool(
    name="General Tool",
    func=general,
    description="Handles general queries using Google's Gemini model",
)

###### AGENT-2 Math Operations ######

def math_func(input_text):
    try:
        result = sympify(input_text)
        return f"The result of {input_text} is {result}"
    except Exception:
        return "Invalid mathematical expression. Please try again."

math_tool = Tool(
    name="Math Tool",
    func=math_func,
    description="Solves mathematical expressions safely",
)

###### AGENT-3 Trivia Agent ######

def facts_func(_):
    facts_list = [
        "The shortest war in history lasted for only 38 minutes between Britain and Zanzibar on August 27, 1896.",
        "The first computer virus was created in 1983 and was called the 'Elk Cloner'.",
        "Honey never spoils; archaeologists found 3,000-year-old honey still edible.",
        "Bananas are berries, but strawberries aren’t.",
        "A day on Venus is longer than a year on Venus.",
        "Octopuses have three hearts and blue blood.",
        "There are more stars in the universe than grains of sand on all the Earth's beaches.",
        "A single strand of spaghetti is called a 'spaghetto'."
    ]
    return random.choice(facts_list)

facts_tool = Tool(
    name="Facts Tool",
    func=facts_func,
    description="Provides random fun facts",
)

# Conversation Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,
    return_messages=True,
)

# Conversation Agent with Tools
conversation_agent = initialize_agent(
    agent="conversational-react-description",
    tools=[general_tool, math_tool, facts_tool],
    llm=model,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
)

# Streamlit UI
st.title("Multi-Agent Chatbot")
st.markdown("### Ask me about:")
st.markdown("- General Queries\n- Mathematical Operations\n- Fun Facts")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    response = conversation_agent.invoke(user_input)
    if isinstance(response, dict) and 'output' in response:
        response = response['output']
    st.session_state["messages"].append({"role": "bot", "content": response})

for msg in st.session_state["messages"]:
    st.write(f"**{msg['role'].capitalize()}**: {msg['content']}")

#import all the required packages

import os
import json
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import random 

os.environ['GOOGLE_API_KEY'] = "AIzaSyBzovUvxbHB7sF09ivq_wLM-g6tB3bOOsg"

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")

###### AGENT-1 General Query Agent ######

def general(input=""):
  try:
    results = model.invoke(input)
    print(results.content)
    return results.content
  except Exception as e:
    return "Sorry, I am not able to understand your query. Please try again."

general_tool = Tool(
  name="General Tool",
  func=general,
  description="Handles general queries using Google's Gemini model",
  input_type="text",
)

###### AGENT-2 Math Operations ######

def math_func(input_text):
  try:
    result = eval(input_text)
    return f"The result of {input_text} is {result}"
  except Exception as e:
    return "Sorry, I am not able to understand your query. Please try again."

math_tool = Tool(
  name="Math Tool",
  func=math_func,
  description="Handles mathematical operations such as solving expressions",
)

###### AGENT-3 Trivia Agent ######

def facts_func(input_text):
  facts_list = [
    "The shortest war in history lasted for only 38 minutes between Britain and Zanzibar on August 27, 1896.",
    "The first computer virus was created in 1983 and was called the 'Elk Cloner'.",
    "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.",
    "Bananas are berries, but strawberries aren't.",
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

#create conversation memory

memory = ConversationBufferWindowMemory(
  memory_key="chat_history",
  k=3,
  return_messages=True,
)

# Conversation Agent with all tools and memory

conversation_agent = initialize_agent(
  agent="chat-conversational-react-description",
  tools=[general_tool, math_tool, facts_tool],
  llm=model,
  verbose=True,
  max_iterations=3,
  early_stopping_method="generate",
  memory=memory,
)

st.title("Multi-Agent Chatbot")

st.markdown("### Ask me About:")
st.markdown("""
- General Queries
- Mathematical Operations 
- Fun Facts
""")

if "messages" not in st.session_state:
  st.session_state.messages = []

user_input = st.text_input("You: ", "")
if user_input:
  st.session_state.messages.append(f"You: {user_input}")
  response = conversation_agent.invoke(user_input)
  if isinstance(response, dict) and 'output' in response:
    response = response['output']
  st.session_state.messages.append(f"Bot: {response}")

for message in st.session_state.messages:
  st.write(message)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.messages import HumanMessage
import datetime


load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")


# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are expert AI researcher.
          Current time: {time}
          1. {first_instruction}
          2. Reflect and critique your answer. Be severe to maximize improvement.
          3. List 1-3 search queries for researching improvements.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~50 word answer."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools([AnswerQuestion], tool_choice="AnswerQuestion")




# Revisor Chain
revise_instructions = """
Revise your previous answer using the new information.
You must retain all fields:
- answer (~50 words)
- reflection (with 'missing' and 'superfluous')
- search_queries (1-3 queries)
- references (list of URLs)
You must include numerical citations [1], [2] in the answer body.
Add a "References" section at the bottom (not counted in word limit).
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools([ReviseAnswer], tool_choice="ReviseAnswer")


# response = first_responder_chain.invoke({
#   "messages": [HumanMessage(content="Write a blog post on Varanasi")]
# })

# print(response)

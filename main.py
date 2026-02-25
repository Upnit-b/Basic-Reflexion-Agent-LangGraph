from typing import List
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

MAX_ITERATIONS = 2


def main():
    graph = StateGraph(MessagesState)

    graph.add_node("draft", first_responder_chain)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revisor", revisor_chain)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "execute_tools")
    graph.add_edge("execute_tools", "revisor")

    def event_loop(state: MessagesState) -> MessagesState:
        messages = state["messages"]
        count_tool_visits = sum(isinstance(m, ToolMessage)
                                for m in messages)
        num_iterations = count_tool_visits
        if num_iterations > MAX_ITERATIONS:
            return "end"
        return "execute_tools"

    graph.add_conditional_edges("revisor", event_loop, {
                                "end": END, "execute_tools": "execute_tools"})

    app = graph.compile()

    response = app.invoke({"messages": HumanMessage(
        content="Write about how small business can leverage AI to grow")})

    print(response)


if __name__ == "__main__":
    main()

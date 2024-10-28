from typing import Callable, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from rich import print as rprint
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def web_search(query: str) -> str:
    """
    Generates context for a query by:
    1. Breaking down the query into additional sub-queries if needed.
    2. Performing web searches for each sub-query.
    3. Returning the combined context without answering the query itself.
    """
    # Step 1: Generate sub-queries to gather context
    prompt = (
        f"You are tasked with gathering context for the following query: {query}. "
        "Generate any additional questions or sub-queries that may help provide context, separated by a single comma with no spaces. "
        "If you don't need any additional sub-queries, just return the original query."
        "Only return queries to help find information, not to answer the question."
    )

    generated_query = llm.invoke(prompt)
    rprint("Generated sub-queries:")
    rprint(generated_query.content)

    search = TavilySearchResults(
        max_results=2,  # Increase the number of results for better context
        description='tavily_search_results_json(query="the search query") - a search engine.',
    )

    # Step 2: Retrieve context for each sub-query
    context = ""
    for sub_query in generated_query.content.split(","):
        sub_query = sub_query.strip().strip('"')
        
        search_results = search.invoke({"query": query.strip().strip('"')})
        rprint(f"Search result for the query: {query} is...")
        rprint(search_results)
        
        if search_results:
            rprint(f"Search results for sub-query '{sub_query}':")
            for result in search_results:
                context += result["content"] + "\n"
                rprint(result["content"])

    # Return the combined context for the original query
    return context

llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools([web_search])
class State(TypedDict):
    messages: Annotated[List, add_messages]
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[web_search])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

if __name__ == "__main__":
    query = "Tell me about the cna article: The Big Read: When home is where the hospital bed is."

    events = graph.stream(
        {"messages": [("user", query)]}, stream_mode="values"
    )
    for event in events:
        last_message = event["messages"][-1]
        rprint(last_message)

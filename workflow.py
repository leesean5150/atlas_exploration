from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_agent: str
    agent_history: List[str]
    context_store: Dict[str, Any]
    final_answer: str | None
    task_status: str
    original_query: str

def create_web_search_agent():
    llm = ChatOpenAI(model="gpt-4-turbo")
    search = TavilySearchResults(
        max_results=2,
        description='tavily_search_results_json(query="the search query") - a search engine.'
    )
    
    def web_search_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        context = state["context_store"]
        query = state["original_query"]
        
        # Generate sub-queries
        sub_query_prompt = (
            f"You are tasked with answering the following: {query}. "
            "Generate more questions you may need answers to in order to answer this questions and return all separated only by a single comma with no spaces."
            "If you do not need anymore context, just return the query."
            "This sub-query will be passed to a search engine to provide more context to answer this query."
            "Ensure that you are using information that is the most up to date."
            "If you are unsure, just return the query. Do not try to guess."
        )
        
        generated_queries = llm.invoke(sub_query_prompt)
        
        # Collect context from searches
        search_context = ""
        for sub_query in generated_queries.content.split(","):
            search_results = search.invoke({"query": sub_query.strip().strip('"')})
            if search_results:
                for result in search_results:
                    search_context += (result['content']) + "\n"
        
        # Generate final answer using collected context
        final_prompt = (
            f"Context:\n{search_context}\n\n"
            f"Question:\n{query}\n"
            "Please answer the question based on the provided context."
        )
        result = llm.invoke(final_prompt)
        
        # Update context store with search results and answer
        updated_context = {
            **context,
            "search_context": search_context,
            "search_result": result.content
        }
        
        return {
            **state,
            "messages": [*messages, AIMessage(content=result.content)],
            "context_store": updated_context,
            "current_agent": "orchestrator"
        }
    
    return web_search_node

def create_orchestrator_agent():
    orchestrator_prompt = """You are the orchestrator agent responsible for:
    1. Analyzing the query to determine which specialist agent to consult
    2. Maintaining context and deciding when to revisit agents for more information
    3. Determining when enough information has been gathered to provide a final answer
    
    Available agents:
    - web_search: Performs intelligent web searches to gather information
    - writer: Synthesizes information into coherent text
    - editor: Refines and improves content
    
    Based on the current state and agent responses, decide the next step."""
    
    orchestrator = ChatOpenAI(temperature=0).bind(
        system_message=orchestrator_prompt
    )
    
    def should_continue(state: AgentState) -> bool:
        messages = state["messages"]
        response = orchestrator.invoke([
            *messages,
            HumanMessage(content="Do we have enough information to provide a final answer? Respond with 'yes' or 'no'.")
        ])
        return response.content.lower().strip() == "no"
    
    def orchestrator_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        context = state["context_store"]
        
        analysis_prompt = f"""
        Current context: {context}
        Agent history: {state['agent_history']}
        
        Determine the next best step:
        1. Which agent should handle the next step?
        2. What specific information do we need from them?
        3. Should we collect more information or finalize the answer?
        
        Respond with either:
        - 'AGENT: [agent_name]' to route to another agent
        - 'FINALIZE' to provide final answer"""
        
        response = orchestrator.invoke([
            *messages,
            HumanMessage(content=analysis_prompt)
        ])
        
        if "FINALIZE" in response.content:
            final_response = orchestrator.invoke([
                *messages,
                HumanMessage(content="Please provide the final comprehensive answer based on all collected information.")
            ])
            
            return {
                **state,
                "current_agent": "END",
                "final_answer": final_response.content,
                "task_status": "complete"
            }
        else:
            next_agent = response.content.split("AGENT:")[1].strip().lower()
            
            return {
                **state,
                "current_agent": next_agent,
                "agent_history": [*state["agent_history"], next_agent],
                "task_status": "in_progress"
            }
    
    return orchestrator_node, should_continue

def create_writer_agent():
    writer = ChatOpenAI(temperature=0.7)
    
    def writer_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        context = state["context_store"]
        
        # Generate content based on search results
        response = writer.invoke([
            *messages,
            HumanMessage(content=f"Using this research: {context.get('search_result', '')}, generate comprehensive content.")
        ])
        
        updated_context = {
            **context,
            "written_content": response.content
        }
        
        return {
            **state,
            "messages": [*messages, response],
            "context_store": updated_context,
            "current_agent": "orchestrator"
        }
    
    return writer_node

def create_editor_agent():
    editor = ChatOpenAI(temperature=0)
    
    def editor_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        context = state["context_store"]
        
        response = editor.invoke([
            *messages,
            HumanMessage(content=f"Review and improve this content: {context.get('written_content', '')}")
        ])
        
        updated_context = {
            **context,
            "edited_content": response.content
        }
        
        return {
            **state,
            "messages": [*messages, response],
            "context_store": updated_context,
            "current_agent": "orchestrator"
        }
    
    return editor_node

def create_workflow_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("orchestrator", create_orchestrator_agent()[0])  # Only use the node function
    workflow.add_node("web_search", create_web_search_agent())
    workflow.add_node("writer", create_writer_agent())
    workflow.add_node("editor", create_editor_agent())

    # Define the routing function for the orchestrator
    def router(state: AgentState) -> str:
        return state["current_agent"]
    
    # Add edges from orchestrator
    workflow.add_edge("orchestrator", router)
    
    # Add edges from other agents back to orchestrator
    for agent in ["web_search", "writer", "editor"]:
        workflow.add_edge(agent, "orchestrator")
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    return workflow.compile()

if __name__ == "__main__":
    workflow = create_workflow_graph()
    result = workflow.invoke({
        "messages": [HumanMessage(content="Who is Leo Messi and what is his father's name and age?")],
        "current_agent": "orchestrator",
        "agent_history": [],
        "context_store": {},
        "final_answer": None,
        "task_status": "started",
        "original_query": "Who is Leo Messi and what is his father's name and age?"
    })
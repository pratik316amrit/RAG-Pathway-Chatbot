import os
import streamlit as st
from pprint import pprint
import sys
import io
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.callbacks import get_openai_callback
import matplotlib.pyplot as plt
import json
import os
import re
import logging
from pprint import pprint
from typing import List

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print(os.environ["OPENAI_API_KEY"])
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


load_environment_variables()

from agents.Chart_Agent import chart_generator
from agents.SQLAgent import SQLAgent
from agents.TableMaker import TableMaker
from agents.FDD_Agents import executive_agent_mode,key_metrics_agent_mode,business_model_agent_mode
from agents.FDD_generator import FDD_Generator_handeler
from agents.visual_json import get_json
from agents.Disc_reframer import Disc_question_reframer
from agents.answer_aggregator import aggregator
from agents.answer_grader import answer_grader
from agents.document_relevent_router import relevency_router
# from agents.finance_react_agent import finance_react_agent
from agents.finance_agent import finance_agent
from agents.hallucination_grader import hallucination_grader
from agents.main_router import question_router
from agents.query_decomposition import decomposer
from agents.question_rewritter import question_rewriter
from agents.reasoning_agent import reasoner
from agents.rectifier import rectifier
from agents.retrieval_grader import retrieval_grader
from agents.verification_agent import verifier
from langchain.schema import Document
from langgraph.graph import END, START, StateGraph
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from rag.rag import (
    compress_documents,
    create_compressor,
    create_groq_llm,
    create_openai_llm,
    create_pathway_client,
    create_prompt_template,
    get_answer,
    retrieve_relevant_documents,
)
from tools.bar_maker import generate_bar_chart
from tools.line_maker import generate_line_chart
from tools.pie_maker import generate_pie_chart
from tools.tools import web_search_tool
from typing_extensions import TypedDict

llm = create_openai_llm()
prompt = create_prompt_template()
BASELINE_VERIFICATION_QUESTIONS = []

# need to add this as a state value in workflow, havent done it yet
NOT_SUPPORTED_COUNTER = 0

# Redirecting stdout to capture print statements (for logging)
class StreamToLogger(io.StringIO):
    def __init__(self):
        super().__init__()
        self.log = ""
    
    def write(self, message):
        self.log += message + "\n"  # Add newline to each log message for clarity
        sys.__stdout__.write(message)  # Also write to the original stdout
    
    def get_logs(self):
        logs = self.log.strip()  # Remove any leading/trailing whitespace
        self.clear_logs()  # Clear logs after retrieval to prevent duplication
        return logs
    
    def clear_logs(self):
        self.log = ""  # Clear stored logs

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)    

logger_stream = StreamToLogger()
sys.stdout = logger_stream  # Redirects print statements to this logger

        
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        max_revision: int
    """

    question: str
    generation: str
    documents: List[str]
    revision_number: int
    max_revisions: int
    final_generation: str

tokens=0
inp_tok=0

async def retrieve(state):
    """
    Retrieve documents based on the current state.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with the key "documents" containing retrieved and compressed documents.
    """
    if "revision_number" not in state or "question" not in state:
        raise ValueError("State must contain 'revision_number' and 'question' keys.")

    question = state["question"]

    # Initialize token tracking
    global tokens
    global inp_tok
    tok = 0
    pr_tok = 0
    try:

        # if state.get("revision_number", 0) == 0:
        print("---RETRIEVE---")
        #print("Question",state["question"])
        # Retrieval
        retriever = await create_pathway_client()
        compressor = create_compressor()

        tok, pr_tok, compressed_docs = compress_documents(retriever, question, compressor)
        tokens += tok
        inp_tok += pr_tok
        print("Len Docs:",len(compressed_docs))
        # if len(compressed_docs)==0:
        #     state["revision_number"] + 1
        #     return web_search(state)
        if not compressed_docs:
            print("No documents retrieved or compressed.")

        return {
            "documents": compressed_docs,
            "question": question,
            # "revision_number": state["revision_number"] + 1,
            "revision_number": state["revision_number"],
            "max_revisions": 3,
        }
        # elif state["revision_number"] > 0:
        #     retriever = await create_pathway_client()
        #     relevant_docs = retrieve_relevant_documents(retriever, question)
        #     # print("RETRIEVE-RELEVANT: ", len(relevant_docs), "---", relevant_docs)
        #     if not relevant_docs:
        #         print("No documents retrieved.")
        #     state["documents"].extend(relevant_docs)
        #     return {
        #         "documents": state['documents'],
        #         "question": question,
        #         # "revision_number": state["revision_number"] + 1,
        #         "revision_number": state["revision_number"],
        #         "max_revisions": 3,
        #     }

    except:
        #print("RETRIEVE-WEB")
        return web_search(state)
    
    # else:
    #     print("---Dynamic-RETRIEVE---")

    #     if "documents" not in state:
    #         raise ValueError("State must contain 'documents' for dynamic retrieval.")

    #     # Dynamic retrieval based on pre-retrieved documents
    #     retriever = create_pathway_client(k=1)
    #     compressor = create_compressor()
    #     pre_retrieved_docs = [doc.page_content for doc in state["documents"]]
    #     updated_docs = state["documents"].copy()

    #     for doc in pre_retrieved_docs:
    #         context_guided_query = f"{question}\ncontext: {doc}"
    #         tok, pr_tok, additional_docs = compress_documents(retriever, context_guided_query, compressor)
    #         tokens += tok
    #         inp_tok += pr_tok
    #         updated_docs.extend(additional_docs)

    #     if not updated_docs:
    #         print("No dynamic documents retrieved or compressed.")

    #     return {
    #         "documents": updated_docs,
    #         "question": question,
    #         "revision_number": state["revision_number"],
    #         "max_revisions": 3,
    #     }


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    print("Question",state["question"])
    global tokens, inp_tok
    
    question = state["question"]
    documents = state["documents"]
    print(f"In GETANSWER {documents}")

    # Get answer and update token counts
    # final_answer='\n'.join([doc.page_content for doc in documents])
    final_answer_pairs = "###Document::\n".join([f"content: {doc.page_content} || metadata:{doc.metadata}" for doc in documents])
    tok, pr_tok, generation = get_answer(final_answer_pairs, question, llm, prompt)
    tokens += tok
    inp_tok += pr_tok
    
    print(f"Total tokens: {tok}")
    print(f"Prompt tokens: {pr_tok}")

    return {
        "documents": documents,
        "question": question,
        "final_generation": generation,
        "revision_number": state["revision_number"],
        "max_revisions": 3,
    }



def grade_documents(state):
    global tokens
    global inp_tok
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    print("Question",state["question"])
    question = state["question"]
    documents = state["documents"]

    # Score each document
    filtered_docs = []
    for doc in documents:
        with get_openai_callback() as cb:
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        print(cb.total_tokens)
        print(cb.prompt_tokens)
        
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    if state["revision_number"] == state["max_revisions"]:
        print("----MAXIMUM REVISIONS REACHED, NOT FILTERING DOCUMENTS----")
        return {"documents": documents, "question": question}

    return {
        "documents": filtered_docs,
        "question": question,
        "revision_number": state["revision_number"],
        "max_revisions": 3,
    }


def transform_query(state):
    global tokens
    global inp_tok
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    print("Question",state["question"])
    question = state["question"]
    documents = state["documents"]

    # Re-write the question
    with get_openai_callback() as cb:
        better_question = question_rewriter.invoke({"question": question})
        state["revision_number"] += 1

    # Update token counts
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens

    print(f"Total tokens: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")

    return {
        "documents": documents,
        "question": better_question,
        "revision_number": state["revision_number"],
        "max_revisions": 3,
    }


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    print("Question",state["question"])
    question = state["question"]
    state["documents"]=state.get("documents",[])
    
    # Perform web search
    search_results = web_search_tool.invoke({"query": question})
    
    # Combine search results into a single document
    try:
        combined_results = "\n".join([result["content"] for result in search_results])
    except Exception as e:
        print(f"Error combining search results: {e}")
        print("Search results: ", search_results)
        combined_results = ""
    web_document = Document(page_content=combined_results, metadata={"source": "web_search"})
    state["documents"].extend([web_document])
    # print("DOCUMENTS-WEB_SEARCH",state["documents"])
    return {
        "documents": state["documents"], 
        "question": question,
        "revision_number": state["revision_number"],
        "max_revisions": 3
    }


### Edges ###


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    print("Question",state["question"])
    global tokens, inp_tok
    question = state["question"]

    # Determine routing using question_router with token tracking
    with get_openai_callback() as callback:
        routing_result = question_router.invoke({"question": question})
        
        # Update token counters
        tokens += callback.total_tokens
        inp_tok += callback.prompt_tokens
        
        # Log token usage
        print(f"Total tokens: {callback.total_tokens}")
        print(f"Prompt tokens: {callback.prompt_tokens}")

    # Route based on determined data source
    if routing_result.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    else:  # vectorstore case
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_retrieve(state):
    """
    Determines whether to retrieve a question (compressed or without compression handled inside retriever) 
    or web-search
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    print("---Decide To Review---")
    print("Question",state["question"])
    if state["revision_number"] == 0 or state["revision_number"] == 1:
        return "retrieve"
    else:
        return "web_search"
    
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, re-generate a question, 
    or route to financial or SQL agents based on the document structure.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    global tokens
    global inp_tok
    print("---ASSESS GRADED DOCUMENTS---")
    print("Question",state["question"])
    question = state["question"]
    filtered_documents = state["documents"]
    
    with get_openai_callback() as cb:
        answerability = relevency_router.invoke({"question": question, "context": filtered_documents})
    
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    
    print(cb.total_tokens)
    print(cb.prompt_tokens)
    
    st.write("###filered docs: ",filtered_documents)
    if filtered_documents and answerability != "answerable":
        print("---DECISION: REDIRECTING TO FINANCE AGENT ---")
        return "not_answerable"

    # Check if all documents are filtered out and if more revisions are allowed
    if not filtered_documents and state["revision_number"] < state["max_revisions"]:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"

    # Check for multiple tables in "text_as_html"
    
    total_table_count = 0
    for doc in filtered_documents:
        # st.write(doc)
        if "text_as_html" in doc:
            soup = BeautifulSoup(doc["text_as_html"], "html.parser")
            total_table_count += len(soup.find_all("table"))

    if total_table_count >= 2:
        print("---DECISION: MULTIPLE TABLES DETECTED, ROUTING TO SQL_AGENT---")
        return "sql_agents"

    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_v_documents_and_question(state):
    global tokens
    global inp_tok
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    print("Question",state["question"])
    global tokens, inp_tok
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check for hallucinations
    with get_openai_callback() as cb:
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        hallucination_grade = score.binary_score
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    print(cb.total_tokens)
    print(cb.prompt_tokens)

    if hallucination_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        # Check if the generation answers the question
        print("---GRADE GENERATION vs QUESTION---")
        with get_openai_callback() as cb:
            score = answer_grader.invoke({"question": question, "generation": generation})
            answer_grade = score.binary_score
        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        print(cb.total_tokens)
        print(cb.prompt_tokens)

        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if state["revision_number"] >= state["max_revisions"]:
                print("---DECISION: MAX REVISIONS REACHED, STOPPING---")
                return "stop"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
    else:
        if state["revision_number"] >= state["max_revisions"]:
            print("---DECISION: MAX REVISIONS REACHED, STOPPING---")
            return "stop"
        else:
            global NOT_SUPPORTED_COUNTER
            NOT_SUPPORTED_COUNTER += 1
            if NOT_SUPPORTED_COUNTER >= 3:
                print("---DECISION: TOO MANY UNSUPPORTED GENERATIONS, STOPPING---")
                return "stop" #We changed it because as sosn as main roter routes a query to web_search and the retrieved content doesn't satisfy for 3 continious web search rather than going to financial agent it stops and generate irrelevant result. 
                # return "financial_agent"
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRYING---")
            return "not supported"


def finance_tool_agent(state):
    """
    Call finance agent tools.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---FINANCE AGENT---",state["question"])
    question = state["question"]
    documents = state["documents"]
    global tokens
    global inp_tok

    # Invoke the finance react agent
    # with get_openai_callback() as cb:
    #     generation = finance_react_agent.invoke({"input": question})
    
    # # Update token counts
    # tokens += cb.total_tokens
    # inp_tok += cb.prompt_tokens
    # print(cb.total_tokens)
    # print(cb.prompt_tokens)
    
    # print("Finance react agent generation: ", generation)
    
    # # Extract the final generation output
    # final_generation = generation["output"]
    
    # Invoke the finance agent
    with get_openai_callback() as cb:
        generation = finance_agent.invoke(question)
        for gen in generation:
            # print("GEN",gen)
            try:
                if "Will be right back" in gen['output'][0][0]:
                  gen['output']=web_search(state)["documents"][-1].page_content
            except Exception as e:
                pass
            try:
                if "Failed to get" in gen['output']:
                  gen['output']=web_search(state)["documents"][-1].page_content
            except Exception as e:
                pass
        # print("GENERATION",generation)
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    print(cb.total_tokens)
    print(cb.prompt_tokens)
    
    
    final_generation = [item['output'] for item in generation]

    if len(final_generation)==0:
        final_generation = [web_search(state)["documents"][-1].page_content]
        
    final_generation_string=f"{final_generation}"
    # print("QUESTION",state["question"],"DOCUMENTS",final_generation_string)
    # print("Documents: ", documents)
    documents.append(Document(page_content=final_generation_string, metadata={"source":"finance_agent_output"}))
    return {
        "documents": documents,
        "question": question,
        "generation": final_generation_string,
        "revision_number": state["revision_number"], 
        "max_revisions": 3
    }


def chart_Agents(generation):
    print("---CHART CREATING AGENT---")

    # Generate the chart reasoning response
    chart_data_response = chart_generator.invoke({"generation": generation})
    print(f"Chart Data Response: {chart_data_response}")

    # Check if the response indicates no chart is possible
    if "No chart possible" in chart_data_response:
        print("No chart can be generated from this context.")
        return "No chart created."

    try:
        # Extract the JSON from the response using regex
        json_match = re.search(r"\{.*\}", chart_data_response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in the response.")
        
        json_text = json_match.group(0)  # Extract the JSON string
        chart_data = json.loads(json_text)  # Parse the JSON string

        # Determine chart type and delegate to the appropriate function
        chart_type = chart_data.get("chartType", "").lower()

        # Generate a unique file path
        save_dir = "charts"
        os.makedirs(save_dir, exist_ok=True)

        # Check if file already exists and create a unique name
        base_filename = "chart.png"
        save_path = os.path.join(save_dir, base_filename)
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(save_dir, f"chart_{counter}.png")
            counter += 1

        if chart_type == "bar":
            return generate_bar_chart(chart_data, save_path)
        elif chart_type == "line":
            return generate_line_chart(chart_data, save_path)
        elif chart_type == "pie":
            return generate_pie_chart(chart_data, save_path)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    except json.JSONDecodeError as jde:
        print(f"JSON parsing error: {jde}")
        return "Failed to create chart due to JSON parsing error."
    except Exception as e:
        print(f"Error in chart generation: {e}")
        return "Failed to create chart."


def reasoning_agent(state):
    """
    Reason based on the question and documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---REASON AGENT---")
    print("Question",state["question"])
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    global tokens, inp_tok

    with get_openai_callback() as cb:
        reasoning_output = reasoner.invoke({"question": question, "documents": state['documents']})
        # print(f"Reasoning Output: {reasoning_output}")
        # print("Documents: ", documents)
        new_document = Document(page_content=reasoning_output, metadata={"source":"reasoning_output"})
        state['documents'].append(new_document)

        if any(keyword in reasoning_output.lower() for keyword in ["chart", "graph", "visualize", "plot"]):
            print("Reasoning indicates a chart may need to be created. Trying to create the chart\n")
            save_path = chart_Agents(reasoning_output)
            if save_path is not None:
                print(f"Saved the image in: {save_path}")

    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    print(cb.total_tokens)
    print(cb.prompt_tokens)

    return {
        "documents": state['documents'],
        "question": question,
        "revision_number": state["revision_number"],
        "max_revisions": 3
    }


def sql_agents(state):
    """
    Handle SQL-based questions by processing metadata and querying the database.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Updated state with the SQL query result.
    """
    print("---SQL QUERY---")
    print("Question",state["question"])
    # Process metadata and populate the database
    chunk = state["document"]
    table_maker = TableMaker(db_name="testDB.db")
    table_maker.process_chunk(chunk)

    # Initialize the SQLAgent
    sql_agent = SQLAgent(
        api_key=os.getenv["GROQ_API_KEY"],
        db_path="example.db",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Ask the SQLAgent for the answer
    question = state["question"]
    answer = sql_agent.ask_question(question)
    answer_string=f"{answer}"
    # print(f"SQL Query Result: {answer}")
    state["documents"].append(Document(page_content=answer_string), metadata={"source":"sql_agent_output"})
    return state


workflow = StateGraph(GraphState)

# Define the nodes (same as provided)
workflow.add_node("web_search", web_search)  
workflow.add_node("retrieve", retrieve)  
workflow.add_node("grade_documents", grade_documents)  
workflow.add_node("generate", generate)  
workflow.add_node("transform_query", transform_query)
workflow.add_node("finance_agent", finance_tool_agent)  
workflow.add_node("reasoning_agent", reasoning_agent)  
workflow.add_node("sql_agents", sql_agents)

# Build graph (same as provided)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
# Directly add an edge from START to the "retrieve" node
# workflow.add_edge(START, "retrieve")

# workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents") #Based on web search it oftent directly generates output without financial tools
workflow.add_edge("web_search", "finance_agent")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "not_answerable": "finance_agent",
        "generate": "generate",
        "sql_agents" : "sql_agents",
    },
)

workflow.add_edge("finance_agent", "reasoning_agent")
workflow.add_edge("reasoning_agent", "generate")
workflow.add_conditional_edges(
    "transform_query",
    decide_to_retrieve,
    {
        "retrieve": "retrieve",
        "web_search": "web_search",
    },
)

workflow.add_conditional_edges(
    "sql_agents",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "stop": END,
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "stop": END,
    },
)



# Compile
app = workflow.compile()

from PIL import Image as PILImage
import io
import asyncio
import os 

output_dir =  os.path.join(os.getcwd()  ,"Pathway_chatbot")
os.makedirs(output_dir, exist_ok=True)

try:
    image_data = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    image = PILImage.open(io.BytesIO(image_data))
    image.save("workflow.png")
except:pass


# Streamlit Interface
st.title("Query Processing Interface")
st.sidebar.title("Logs")



query_input = st.text_area("Enter your query", value="You can ask a question here.")
run_button = st.button("Run")
org_input = st.text_area("Organization", value="Only enter the name of the organization")
generate_FDD = st.button("Generate FDD Report")


async def Generator(subQueries, verification=" "):
    
    generations = []  # To store all generations

    async def process_subquery(sub_query):
        st.write(f"#### Processing {verification} query: {sub_query}")
        st.write("-" * 50)

        count = 0
        # Run the streaming process for each query
        async for output in app.astream({"question": sub_query, "revision_number": 0, "max_revisions": 3}):
            # Capture output generation steps
            for key, value in output.items():
                st.write(f"#### At Node {key}")

            logs = logger_stream.get_logs()
            if logs:
                st.sidebar.text_area("Logs", logs, height=200, max_chars=None, key=f"{sub_query}_{count}")

            # if "generation" in value:
            #     return {"query": sub_query, "generation": value["generation"]}
            count += 1
        return {"query": sub_query, "generation": value["final_generation"]}

    tasks = [process_subquery(sub_query) for sub_query in subQueries]
    generations = await asyncio.gather(*tasks)

    return generations


global data

data = {
    "question_by_key": """ """,
    "question_by_business": """ """,
    "question_by_executive": """ """,
    "answer_for_key": """ """,
    "answer_for_business" : """ """,
    "answer_for_executive" : """ """,
    "agent_used" : """""",
    "suggestion_by_key" : """ """,
    "suggestion_by_business" : """ """,
    "suggestion_by_executive": """ """,
    "Is_key_metrics_satisfied" : "0",
    "Is_business_metrics_satisfied" : "0",
    "Is_executive_metrics_satisfied" : "0"
}

filename = "text_output.txt"
filename2 = "gen_output.txt"
global FDD_rev
FDD_rev = 0

Rev_limit = 2
from tools.helper import append_to_file

def getMajorityVote(data):
    keys_to_check = [
        "Is_key_metrics_satisfied",
        "Is_business_metrics_satisfied",
        "Is_executive_metrics_satisfied"
    ]

    unsatisfied_count = sum(1 for key in keys_to_check if data.get(key) == "0")

    return unsatisfied_count

def Discussion(company,data,key_agent,executive_agent,business_agent):
    global FDD_rev
    prompt = f"There are questions asked by each agent and their respective answers in {data}. Discuss among other agents and suggest changes if any"
    # st.write(data)        
    
    key_suggestion,agent_used = key_agent(mode="Disc")
    data["suggestion_by_key"] = key_suggestion.invoke(prompt)

    first_line = data["suggestion_by_key"].splitlines()[0].strip()
    # st.write(first_line)
    if "Not Satisfied" in first_line:
        data["Is_key_metrics_satisfied"] = "0"
    elif "Satisfid" in first_line:
        data["Is_key_metrics_satisfied"] = "1"
    business_suggestion,agent_used = business_agent(mode="Disc")
    data["suggestion_by_business"] = business_suggestion.invoke(prompt)

    first_line = data["suggestion_by_business"].splitlines()[0].strip()
    # st.write(first_line)
    if "Not Satisfied" in first_line:
        data["Is_business_metrics_satisfied"] = "0"
    elif "Satisfid" in first_line:
        data["Is_business_metrics_satisfied"] = "1"

    executive_suggestion,agent_used = executive_agent(mode="Disc")
    data["suggestion_by_executive"] = executive_suggestion.invoke(prompt)

    first_line = data["suggestion_by_executive"].splitlines()[0].strip()
    # st.write(first_line)
    if "Not Satisfied" in first_line:
        data["Is_executive_metrics_satisfied"] = "0"
    elif "Satisfid" in first_line:
        data["Is_executive_metrics_satisfied"] = "1"


    # Last check to make sure that the final answers are appended
    if FDD_rev >= Rev_limit:
        final_txt = data["answer_for_key"] + "\n" + data["answer_for_business"] + "\n" + data["answer_for_executive"]
        append_to_file(filename,final_txt)
        return
    if getMajorityVote(data) > 1:
        st.write("### More than 2 agents are Not satsified! Running Reframer")
        st.write(data)
        pmt1 = f"""Look at these questions asked by key_metrics_agent in {data["question_by_key"]}, and suggestion by other agents : {data["suggestion_by_key"]}, {data["suggestion_by_business"]}, {data["suggestion_by_executive"]},remake relevant questions for this agent."""
        key_queries_new = Disc_question_reframer.invoke(pmt1)

        pmt2 = f"""Look at these questions asked by business_agent in {data["question_by_business"]}, and suggestion by other agents : {data["suggestion_by_key"]}, {data["suggestion_by_business"]}, {data["suggestion_by_executive"]},remake relevant questions for this agent."""
        business_queries_new = Disc_question_reframer.invoke(pmt2)

        pmt3 = f"""Look at these questions asked by executive_agent in {data["question_by_executive"]}, and suggestion by other agents : {data["suggestion_by_key"]}, {data["suggestion_by_business"]}, {data["suggestion_by_executive"]},remake relevant questions for this agent."""
        executive_queries_new = Disc_question_reframer.invoke(pmt3)

        net_queries_new = [key_queries_new,business_queries_new,executive_queries_new]


        FDD_rev += 1
        st.write("### FDD_REV",FDD_rev)
        process_question(company,net_queries_new,tokens,inp_tok)

    else:
        final_txt = data["answer_for_key"] + "\n" + data["answer_for_business"] + "\n" + data["answer_for_executive"]
        append_to_file(filename,final_txt)


global final_report
final_report = """"""
def process_question(company,netQueries,tokens,inp_tok):
    """
    Process a given question by decomposing it into sub-queries, running the workflow for each sub-query,
    and aggregating the answers to generate a final answer.

    Args:
        question (str): The input question to process.

    Returns:
        None: The function handles displaying results directly in the Streamlit interface.
    """

    if FDD_rev >= Rev_limit:
        st.write("### Rev Limit Hit")
        return
    
    global final_report

    for queries in netQueries:
        logger_stream.clear_logs()  # Clear any previous logs

        # Decompose the input query into sub-queries
        inputs = {"question": queries, "revision_number": 0, "max_revisions": 3}
        st.write("### original Questions: \n",inputs["question"])

        # Initialize the NOT_SUPPORTED_COUNTER
        NOT_SUPPORTED_COUNTER = 0

        with get_openai_callback() as cb:
            subQueries = decomposer.invoke(inputs).sub_queries

        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens

        st.write("### Subqueries:",subQueries)
        # Generate outputs for the sub-queries
        generations = asyncio.run(Generator(subQueries, verification=" "))
        answers = []
        for gen in generations:
            answers.append(gen["generation"])

        # Aggregate answers and display final answer
        if generations:
            with get_openai_callback() as cb:
                ans = aggregator.invoke({"question": queries, "answers": ", ".join(answers)})
                
            tokens += cb.total_tokens
            inp_tok += cb.prompt_tokens
            # reply = key_metrics_handler.invoke(ans.answer)
            # print(reply)
            st.write("### Final Answer:")
            st.write(ans.answer)
        
        # Display each subquery result in a dropdown
        for gen in generations:
            with st.expander(f"Sub-query: {gen['query']}"):
                st.write(gen["generation"])

            if queries == key_queries:
                with open("key_metrics_sub.txt", 'a') as file:
                    file.write("Question :" + gen["query"] + '\n')
                    file.write("Answer: " + gen["generation"] + '\n')  # Append the text followed by a newline
                    st.write("Answers to the key subqueries has been appended\n")

            with open(filename2, 'a') as file:
                file.write("Question :" + gen["query"] + '\n')
                file.write("Answer: " + gen["generation"] + '\n')  # Append the text followed by a newline
                st.write("Answers to the subqueries has been appended\n")

        with open(filename2,'r') as f:
            content = f.read()
            if queries == key_queries:
                save_to_pdf(content,"key_metric.pdf")
                key_generation = FDD_Generator_handeler.invoke(f"Agent is key_metrics and company is {company}" + content)
                final_report = final_report + key_generation + "\n"
            elif queries == business_quries:
                business_generation = FDD_Generator_handeler.invoke(f"Agent is business_agent and company is {company}" + content)
                final_report = final_report + business_generation + "\n"
            elif queries == exec_queries:
                executive_generation = FDD_Generator_handeler.invoke(f"Agent is executive agent and company is {company}" + content)
                final_report = final_report + executive_generation + "\n"

        if os.path.exists(filename2):
            os.remove(filename2)
            st.write(f"{filename2} has been deleted")


        if queries == key_queries:
            data["question_by_key"] = queries
            data["answer_for_key"] = ans.answer
        elif queries == business_quries:
            data["question_by_business"] = queries
            data["answer_for_business"] = ans.answer
        else:
            data["question_by_executive"] = queries
            data["answer_for_executive"] = ans.answer
    
    append_to_file("final_reportt.txt",final_report)

    # gen_rep = FDD_Generator_handeler.invoke(final_report)
    st.write("### FINAL REPORT_ENHANCED:\n",final_report)
    save_to_pdf(final_report,f"{org_input}_report_gen.pdf")
    
    Discussion(company=company,data=data,key_agent=key_metrics_agent_mode,business_agent=business_model_agent_mode,executive_agent=executive_agent_mode)
    st.write(data)
    with open(filename,'r') as f:
        content = f.read()
        Generated_Report = FDD_Generator_handeler.invoke(content)
        Generated_json = get_json(Generated_Report)
        st.write(Generated_json)
    
    st.write(tokens)
    st.write(inp_tok)

    # Display each subquery result in a dropdown
    # for gen in generations:
    #     with st.expander(f"Sub-query: {gen['query']}"):
    #         st.write(gen["generation"])
    # Show final log state
    
    st.sidebar.write(logger_stream.get_logs())



from reportlab.lib.pagesizes import letter


from tools.helper import save_to_pdf


if generate_FDD and org_input:
    # Check if the file exists and delete it
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} already exists. It has been deleted.")


    input_queries101 = f"Ask the relevant Questions to the RAG about {org_input} to produce a relavant Financial Due Deligence Report."
    exec_queries = executive_agent_mode(mode="QnA").invoke(input_queries101)
    business_quries = business_model_agent_mode(mode="QnA").invoke(input_queries101)
    key_queries = key_metrics_agent_mode(mode="QnA").invoke(input_queries101)

    net_queries = [key_queries,exec_queries,business_quries]

    process_question(org_input,net_queries,tokens,inp_tok)

    final_txt = data["answer_for_key"] + "\n" + data["answer_for_business"] + "\n" + data["answer_for_executive"]
    append_to_file(filename,final_txt)
    with open(filename, 'r') as file:  # Open file in read mode
        content = file.read()  # Read the entire content of the file
        Generated_Report = FDD_Generator_handeler.invoke(content)
        st.write(Generated_Report)
        # save_to_pdf(Generated_Report,f"{org_input}_Report.pdf")
        Generated_json = get_json(Generated_Report) # Generated JSON for the cards
        st.write(Generated_json)
    


if run_button and query_input:
    logger_stream.clear_logs()  # Clear any previous logs

    # Decompose the input query into sub-queries
    inputs = {"question": query_input, "revision_number": 0, "max_revisions": 3}
    
    # Initialize the NOT_SUPPORTED_COUNTER
    NOT_SUPPORTED_COUNTER = 0

    with get_openai_callback() as cb:
        subQueries = decomposer.invoke(inputs).sub_queries

    # Track token usage
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens

    st.write("### Sub-queries:", subQueries)

    # Generate outputs for the sub-queries
    generations = asyncio.run(Generator(subQueries, verification=" "))
    answers = []
    for gen in generations:
        answers.append(gen["generation"])

    # Aggregate answers and display final answer
    if generations:
        with get_openai_callback() as cb:
            ans = aggregator.invoke({"question": query_input, "answers": ", ".join(answers)})


        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        st.write("### Final Answer:")
        st.write(ans.answer)
        st.write(tokens)
        st.write(inp_tok)

    # Display each subquery result in a dropdown
    for gen in generations:
        with st.expander(f"Sub-query: {gen['query']}"):
            st.write(gen["generation"])
        
        # Chain of Verification
        
        # baseline_response=ans.answer
        # BASELINE_VERIFICATION_QUESTIONS.extend(verifier.invoke({"query": query_input, "baseline_response":baseline_response}).verification_questions)
        # verification_query_responses = Generator(BASELINE_VERIFICATION_QUESTIONS,verification=" verification ")
        # verified_answers = []
        # for gen in verification_query_responses:
        #     verified_answers.append(gen["generation"])
        # verified_answer = rectifier.invoke({"query":query_input,"baseline_response":baseline_response,"verified_answers":verified_answers}).corrected_baseline
        # st.write("### Verified-Final Answer:")
        # st.write(verified_answer)


    # Show final log state
    
    st.sidebar.write(logger_stream.get_logs())
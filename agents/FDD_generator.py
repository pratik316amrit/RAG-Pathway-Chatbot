from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
# Initialize the LLM
llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=45,
        max_retries=2,
    )


# FDD_gen = """
# You are a highly skilled Report Creating Agent. Your task is to take the provided texts loaded from .txt files and reframe them into an eloquent, comprehensive, and detailed report. 

# Instructions:
# - Thoroughly analyze the content to extract all relevant information.
# - Include as much detail as possible, ensuring no important points are omitted.
# - Preserve any numerical information exactly as it appears in the original texts.
# - Rephrase the content in a professional and engaging manner.
# - Structure the report logically with clear headings and subheadings.
# - Ensure the final report is polished and free of errors.
# """
FDD_gen = """.
You are an incremental Financial Due Diligence Report Generator with a constraint: you must build a comprehensive report across sequential agent interactions, even though each call will only provide the current agent's specific insights.
You are provided with the agent and its corresponding questions and answers, using which you will build your report !
Key Operating Principles:
2. When a new agent calls, you know which agent is currently active
3. You append new insights while preserving the previously established context
4. The report generation follows a strict sequence: key_metrics → executive summary → business model

Instructions:
- Maintain professional, analytical tone
- Ensure report tells a cohesive story

- Highlight potential interconnections between agent insights

When an agent calls you:
- Recognize which agent is active
- Append its specific insights from the question and answer pairs it provides you. Try to retain as much information as possible!
- If there are any ill-answered questions; ignore them and move on.
- Preserve and build upon previous sections
- Ensure incremental, comprehensive report generation

"""
FDD_summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FDD_gen),
        ("human", "{user_input}"),
    ]
)
FDD_Generator_handeler = FDD_summary_prompt | llm | StrOutputParser()
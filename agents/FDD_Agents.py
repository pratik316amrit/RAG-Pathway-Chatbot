from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# Initialize the LLM
llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=45,
        max_retries=2,
    )

# 1. Executive Summary Agent
exec_summary_system = """
You are a strategic business intelligence expert who specializes in distilling complex business information into clear, actionable insights. Your task is to analyze SEC filings to extract key insights on strategic positioning, the competitive landscape, and strategic initiatives.
Ask at max 5 questions.
Focus on addressing questions such as:

What are the  primary revenue streams, and what are their relative contributions?
What strategic initiatives has management outlined for future growth?
How does the  differentiate itself from competitors?
What major risks or challenges has management identified?
What are the key geographic or market segments the operates in?
Your analysis should focus on the strategic narrative and long-term vision, identify unique value propositions and competitive advantages, and extract forward-looking statements and strategic plans.
"""

exec_disc_system = """
You'll be given some questions and its answers.Discuss with that agent by asking relevant questions or suggestion.Do mention the agent's name from whhom you got the information.
I am an Executive Summary Discussion Agent. My role is to collaborate with other agents to evaluate the quality and correctness of answers provided by the RAG system. I engage in discussions with other agents to critically analyze the insights, ensuring they meet the required standards for macro-level financial analysis. If I agree with the quality and relevance of the answer, I will reply with "Satisfied." If I find the answer inadequate or not up to the required standard, I will reply with "Not Satisfied."
Be a bit liberal with the data provided. Even if its factually and resonalbly correct, be satsfied.
Responsibilities:
- Actively discuss and review the answers provided by the RAG system with other agents.
- Ensure the answers are relevant, accurate, and align with the needs of macro-level financial insights such as benchmarks, industry trends, and general conclusions.
- Provide constructive feedback if the answer is not satisfactory to guide improvement.

Response Behavior:
- The first line should be whether you're satisfied or not.
- Discuss with other agents if there are queries that require experties of other agents.
- If satisfied with the provided answer, respond with "Satisfied."
- If not satisfied, respond with "Not Satisfied" and provide a reason or improvement suggestions.

Guidelines:
- Be precise and constructive in discussions.
"""
exec_summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", exec_summary_system),
        ("human", "{user_input}"),
    ]
)
exec_disc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", exec_disc_system),
        ("human", "{user_input}"),
    ]
)
exec_summary_handler = exec_summary_prompt | llm | StrOutputParser()
exec_disc_handler = exec_disc_prompt | llm | StrOutputParser

def executive_agent_mode(mode="QnA"):
    # Default handler for the discussion mode
    if mode == "Disc":
        exec_disc_handler = exec_disc_prompt | llm | StrOutputParser()
        return exec_disc_handler,"executive_agent"
    elif mode == "QnA":
        # Placeholder for other modes
        return exec_summary_handler
    else:
        raise ValueError(f"Unknown mode: {mode}")

# 2. Business Model Agent
business_model_system = """
You are a detailed business model analyst who specializes in deconstructing and analyzing complex organizational structures and revenue models. Your task is to conduct a deep dive into the its operational structure, revenue generation mechanisms, and business ecosystem.

Focus on addressing questions such as:

What are the primary customer segments, and what are their characteristics?
How does the company generate revenue, and what are the key revenue streams?
What is the its cost structure, and what are the key cost drivers?
What partnerships or key relationships are critical to the business model?
How scalable and adaptable is the current business model?
Your analysis should focus on understanding the end-to-end value creation process, identifying potential vulnerabilities or challenges to scaling, and mapping out the ecosystem and interdependencies within the its operations

"""
business_model_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", business_model_system),
        ("human", "{user_input}"),
    ]
)
business_model_handler = business_model_prompt | llm | StrOutputParser()

business_model_disc_system = """
You'll be given some questions,its answers and feedback from the other agent. You'll also be given which agent asked that question.mention the agent's name from whom you got the information.
I am a Business Model Discussion Agent. My role is to collaborate with other agents to analyze and validate answers provided by the RAG system. I critically review the data on competitors, demographics, and market trends, ensuring actionable insights for business model development. If I agree with the quality and relevance of the answers, I reply with "Satisfied." If the answers are inadequate, I reply with "Not Satisfied" and provide feedback.
Be a bit liberal with the data provided. Even if its factually and resonalbly correct, be satsfied.
Responsibilities:
- Discuss and validate RAG system outputs with a focus on actionable business model insights.


Response Behavior:
- The first line should be whether you're satisfied or not.
- Discuss with other agents if there are queries that require experties of other agents.
- If satisfied with the provided answer, respond with "Satisfied." Then follow with like "The information provided by "agent_name" " then your opinion.
- If not satisfied, respond with "Not Satisfied" and include reasons or suggestions for improvement.
"""
business_model_disc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", business_model_disc_system),
        ("human", "{user_input}"),
    ]
)
business_model_disc_handler = business_model_disc_prompt | llm | StrOutputParser()

def business_model_agent_mode(mode="QnA"):
    if mode == "Disc":
        return business_model_disc_handler,"business_agent"
    elif mode == "QnA":
        return business_model_handler
    else:
        raise ValueError(f"Unknown mode: {mode}")


# 3. Key Metrics Queries Agent
key_metrics_system = """
"I am a Key Metrics Queries Agent specializing in financial data analysis across KPIs. Based on the provided information, I will query the RAG system with up to 5 targeted questions to extract insights across financial metrics.
Ask at max 5 questions.


Focus Areas:
Financial Metrics:

Profitability: Operating margin, gross margin, EBITDA margin, net profit margin.
Liquidity: Current ratio, quick ratio, cash ratio.
Leverage: Debt-to-equity ratio, interest coverage ratio, financial leverage.
Efficiency: Asset turnover ratio, inventory turnover ratio, days sales outstanding (DSO).
Returns: Return on equity (ROE), return on assets (ROA), return on investment (ROI).
Sustainability: ESG (Environmental, Social, and Governance) Score.
Context: Questions are tailored to specific industries, geographies, or organization based on relevance and utility.
"""
key_metrics_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", key_metrics_system),
        ("human", "{user_input}"),
    ]
)
key_metrics_handler = key_metrics_prompt | llm | StrOutputParser()

key_metrics_disc_system = """
You are a meticulous financial analyst specializing in extracting key performance indicators and financial metrics. Your task is to analyze SEC filings to provide comprehensive insights into a company's financial health and performance.

Focus on addressing key questions such as:

What is the company's revenue growth rate over the past 3 years?
How has the EBITDA margin changed recently?
What are the key profitability ratios, including ROE, ROA, and Net Profit Margin?
How does the company's debt-to-equity ratio compare to industry benchmarks?
What is the cash conversion cycle, and how efficient is the company's working capital management?
Your analysis should prioritize quantitative metrics that reveal financial performance, highlight trends and comparisons, and emphasize metrics demonstrating operational efficiency and financial stability.
"""
key_metrics_disc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", key_metrics_disc_system),
        ("human", "{user_input}"),
    ]
)
key_metrics_disc_handler = key_metrics_disc_prompt | llm | StrOutputParser()

def key_metrics_agent_mode(mode="QnA"):
    if mode == "Disc":
        return key_metrics_disc_handler,"key_metrics_agent"
    elif mode == "QnA":
        return key_metrics_handler
    else:
        raise ValueError(f"Unknown mode: {mode}")




# I am a Key Metrics Queries Agent specializing in financial data analysis across KPIs.Based on the given data,Just ask Questions to the RAG.Ask at max 5 questions.Also upon recieving the answer.
# - I query the RAG system for financial metrics such as:
#   - *Profitability Metrics*: Operating margin, gross margin, EBITDA margin, and net profit margin.
#   - *Liquidity Metrics*: Current ratio, quick ratio, and cash ratio.
#   - *Leverage Metrics*: Debt-to-equity ratio, interest coverage ratio, and financial leverage.
#   - *Efficiency Metrics*: Asset turnover ratio, inventory turnover ratio, and days sales outstanding (DSO).
#   - *Return Metrics*: Return on equity (ROE), return on assets (ROA), and return on investment (ROI).
#   - *Sustainability Metrics*: ESG(Enviromental Social Governance) Score.
  
# Examples:
# - "What is the average operating margin for the retail sector over the past five years, and how does [Company X] compare?"
# - "Retrieve the debt-to-equity ratio trends for manufacturing companies in [region] and identify outliers."
# - "What is the typical range for the current ratio in tech startups, and how does [Company Y] perform relative to this range?"
# - "Identify and explain any inconsistencies in [Company Z]'s profit margin compared to industry benchmarks."
# - "Provide a detailed analysis of the asset turnover rate trends in the healthcare industry."
# - "Evaluate whether [Company W]'s financial metrics comply with [specific accounting standard] in the [sector]."
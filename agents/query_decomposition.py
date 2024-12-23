### Hallucination Grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

# Data model
class QueryDecomposition(BaseModel):
    """to get List of subqueries"""

    sub_queries: list = Field(
        description="List of sub-queries only decomposed from the original query."
    )


# LLM with function call
# llm = ChatGroq(
#     model="llama-3.2-90b-text-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
structured_llm_grader = llm.with_structured_output(QueryDecomposition)

# Prompt
system = """You are an expert in breaking down complex queries into smaller sub-queries that conveys same meaning as original query.
        Given the query below, decompose it into a list of smaller sub-queries. The smaller sub-queries should be independent of each other, one's output should not depend on the other's output:

        Query: "{{input_query.query}}"

        Decompose it into simpler sub-queries and provide them as a list of strings only.
"""
decomposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Query: \n\n {question}"),
    ]
)

decomposer = decomposition_prompt | structured_llm_grader

# Example
# query = "Can you find companies stock similar to Apple (AAPL) in the US market?"
# ans = decomposer.invoke({"question":query})
# print(ans)

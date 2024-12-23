### Retrieval Grader

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
# llm = ChatGroq(
#     model="llama-3.2-90b-vision-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a good reasoner. Follow the steps below to identify the best context needed to answer a question using the provided information:

1. Step 1 - Carefully analyze the question to identify the main topic or what specific information is being requested.
2. Step 2 - Go through the provided information and assess how each part relates to the question.
3. Step 3 - Determine which pieces of information are directly related to answering the question and which are irrelevant.
4. Step 4 - Choose the context that is most closely aligned with the question, ensuring it contains the key details necessary for a precise answer.

Follow the steps above to identify the best context needed to answer the question without hallucinating.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# resp=retrieval_grader.invoke({"question":"What are the benefits of using green energy in residential homes?","document":"Green energy has become a popular topic in recent environmental discussions, with various nations setting ambitious goals to transition away from fossil fuels. Conferences around the world have highlighted the importance of renewable energy sources in combating climate change. Some regions are investing heavily in wind farms and solar farms to boost the percentage of renewables in their national energy mix.Educational institutions are incorporating renewable energy topics into their curriculums, teaching students about the science behind solar panels, wind turbines, and geothermal energy. The global green energy market is also seeing significant technological advancements, leading to more efficient energy capture and storage solutions. However, the adoption rate of green energy varies widely by region, with some areas more proactive in implementing sustainable solutions than others."})
# print(resp)

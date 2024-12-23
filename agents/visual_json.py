import json
from textwrap import dedent
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI()

MODEL = "gpt-4o-2024-08-06"

dashboard_prompt = '''
    You are an AI trained to provide detailed and structured information in response to business and financial queries. 
    Please provide the requested data in JSON format where applicable, using the appropriate structure for each section.
'''

class KeyMetric(BaseModel):
    p_e_ratio: str
    p_b_ratio: str
    debt_to_equity_ratio: str
    free_cashflow: str
    peg_ratio: str
    working_capital_ratio: str
    quick_ratio: str
    earning_ratio: str
    return_on_equity: str
    esg_score: str
    
class CompanyCapital(BaseModel):
    stakeholder_name1: str
    stakeholder_stocks_value1: str
    stakeholder_equity1: str

    stakeholder_name2: str
    stakeholder_stocks_value2: str
    stakeholder_equity2: str

    stakeholder_name3: str
    stakeholder_stocks_value3: str
    stakeholder_equity3: str
    
class Market(BaseModel):
    country1: str
    market_percentage1: str

    country2: str
    market_percentage2: str

    country3: str
    market_percentage3: str
    
class Management(BaseModel):
    name1: str
    designation1: str
    vision_for_company1: str

    name2: str
    designation2: str
    vision_for_company2: str

    name3: str
    designation3: str
    vision_for_company3: str
    
class SDG(BaseModel):
    sdg_number1: str
    goal_description1: str
    contribution1: str

    sdg_number2: str
    goal_description2: str
    contribution2: str

    sdg_number3: str
    goal_description3: str
    contribution3: str
    



def get_json(context:str):
    answer = []
    
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                Can you provide the following financial metrics:  
                1. Price-to-Earnings (P/E) Ratio  
                2. Price-to-Book (P/B) Ratio  
                3. Debt-to-Equity Ratio  
                4. Free Cash Flow  
                5. Price/Earnings-to-Growth (PEG) Ratio  
                6. Working Capital Ratio  
                7. Quick Ratio  
                8. Earnings Per Share (EPS)  
                9. Return on Equity (ROE)  
                10. ESG Score  
                ''')},
        ],
        response_format=KeyMetric,
    )
    answer.append(completion.choices[0].message.content)
    
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                "Can you provide the capitalization table for the company, including the following details for each shareholder:
                    1. Shareholder name 
                    2. Number of shares they own
                    3. Percentage of ownership 
                ''')},
        ],
        response_format=CompanyCapital,
    )
    answer.append(completion.choices[0].message.content)
    
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                Can you provide the market penetration data for the company in different countries? Include the following details for each country:
                    1. Country name
                    2. Market penetration percentage
                ''')},
        ],
        response_format=Market,
    )
    answer.append(completion.choices[0].message.content)    
    
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                Can you provide details about the management team and their ownership in the company? Include the following information for each member:
                    1.  Name of the person
                    2. Designation (e.g., CEO, CTO, CFO)
                    3. Vision for the company
                ''')},
        ],
        response_format=Management,
    )
    answer.append(completion.choices[0].message.content)
    
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                "Can you provide details about the company's contributions to the UN Sustainable Development Goals (SDGs)? Include the following information for each SDG the company supports:
                    1. SDG number
                    2. Goal description
                    3. Contribution by the company
                ''')},
        ],
        response_format=SDG,
    )
    answer.append(completion.choices[0].message.content)

    return answer



context = '''
    Here’s a set of dummy data in text format for each of the sections based on the previously generated questions:

---

### Capitalization Table

**Capitalization Table:**

- **Founder 1**:  
  Shares: 500,000  
  Ownership: 50%

- **Founder 2**:  
  Shares: 300,000  
  Ownership: 30%

- **Investor A**:  
  Shares: 100,000  
  Ownership: 10%

- **Employee Pool**:  
  Shares: 50,000  
  Ownership: 5%

- **Investor B**:  
  Shares: 50,000  
  Ownership: 5%

---

### Market Penetration

**Market Penetration:**

- **United States**:  
  Market Penetration: 75%

- **Canada**:  
  Market Penetration: 60%

- **Germany**:  
  Market Penetration: 45%

- **India**:  
  Market Penetration: 30%

- **Brazil**:  
  Market Penetration: 40%

- **Japan**:  
  Market Penetration: 50%

- **Australia**:  
  Market Penetration: 70%

- **South Africa**:  
  Market Penetration: 20%

---

### Management Ownership and Vision

**Management Ownership and Vision:**

- **John Doe (CEO)**:  
  Vision: To lead the company to become the global leader in innovative technology, focusing on sustainability and market disruption.

- **Jane Smith (CTO)**:  
  Vision: To create cutting-edge, scalable technologies that redefine user experiences and drive industry growth.

- **Michael Johnson (CFO)**:  
  Vision: To build strong financial foundations and ensure sustainable growth, focusing on long-term profitability and shareholder value.

---

### SDG Contributions

**SDG Contributions:**

- **SDG 2 (Zero Hunger)**:  
  Contribution: Through partnerships with food banks and sustainable farming initiatives, the company helps reduce hunger and food insecurity.

- **SDG 3 (Good Health and Well-being)**:  
  Contribution: The company promotes employee health and well-being with comprehensive health programs and supports access to healthcare in underdeveloped regions.

- **SDG 4 (Quality Education)**:  
  Contribution: The company invests in education by providing scholarships and training programs for underserved communities and employees.

- **SDG 5 (Gender Equality)**:  
  Contribution: The company fosters a diverse and inclusive workplace, actively promoting gender equality in hiring and leadership roles.

- **SDG 7 (Affordable and Clean Energy)**:  
  Contribution: The company invests in renewable energy sources and works on developing energy-efficient products.

- **SDG 9 (Industry, Innovation, and Infrastructure)**:  
  Contribution: The company supports sustainable industry practices, innovation in technology, and infrastructure development in emerging markets.

- **SDG 12 (Responsible Consumption and Production)**:  
  Contribution: The company adheres to sustainable sourcing practices, minimizes waste, and promotes recycling and circular economy initiatives.

- **SDG 13 (Climate Action)**:  
  Contribution: The company has committed to reducing its carbon footprint through energy-efficient operations, carbon offset programs, and sustainable product designs.

- **SDG 17 (Partnerships for the Goals)**:  
  Contribution: The company collaborates with NGOs, governments, and other businesses to achieve SDG targets through shared initiatives and joint projects.


    '''
    
    
# print(get_json(context))
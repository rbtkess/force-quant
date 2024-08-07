from flask import Flask, request, jsonify, Response
from flask_cors import CORS 
from flask_socketio import SocketIO, emit
import time
import operator
from typing import Annotated, Any, Sequence, Dict, Optional, Tuple, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import requests # used to access wikipedia
from bs4 import BeautifulSoup # used to process HTML from wikipedia
from IPython.display import display, HTML, Image
import os # for referencing environment variables
from dotenv import load_dotenv # for loading environment variables 
import yfinance as yf # used to obtain P/E ratios
from tavily import TavilyClient # used to search web for context
from langchain_anthropic import ChatAnthropic 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

model = 'claude-3-5-sonnet-20240620'

app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/process', methods=['POST'])
def process_input():
    data = request.get_json()
    input_text = data.get('input')

    result = graph.invoke({"ticker":input_text}, debug=False)
    result['summary'] = result['summary'].json()

    output = result

    return jsonify(output)

load_dotenv() # load environment variables from .env file

class PortersForceScore(BaseModel):
    '''
    A score for a specific Porters 5 Force, including explaination.
    '''
    force: Literal[
        'Rivalry',
        'New Entrants',
        'Suppliers',
        'Customers',
        'Substitutes'
    ] = Field(
        ...,
        alias='force_name',
        description='''The Porters 5 Force being scored, including:
        - Rivalry: The intensity of competition within the industry.
        - New Entrants: The threat of new companies entering the industry.
        - Suppliers: The bargaining power of suppliers to the industry.
        - Customers: The bargaining power of customers in the industry.
        - Substitutes: The threat of alternative products or services
        '''
        )
    score: float = Field(
        ...,
        alias='force_score',
        description='''
        A score in the range of 0 (worse for the company) to 10 (best for the company).
        The scale should be measured relative to the average company in the S&P 500 index, 
        assuming that the average for all companies is 5.
        '''
        )
    justification: str = Field(
        ...,
        alias='force_justification',
        description='''A 2-3 sentence explaination for how the score was derived.'''
        )
    class Config:
        allow_population_by_field_name = True

class PortersFiveForcesSummary(BaseModel):
    '''
    A summary of Porter's Five Forces analysis, including average score.
    '''
    forces: List[PortersForceScore] = Field(
        ...,
        alias='forces_scores',
        description='''A list of scores for each of the five forces.'''
    )

    @property
    def average_score(self) -> float:
        '''
        Calculate the average score of the provided Porters 5 Forces.
        '''
        total_score = sum(force.score for force in self.forces)
        return total_score / len(self.forces)
    
    class Config:
        allow_population_by_field_name = True

def merge_dictionaries(dict1, dict2):
    '''
        Merges two dictionaries, giving priority to keys and values from dict1.
    '''
    # Start with a copy of dict1 to ensure its keys and values are prioritized
    merged_dict = dict1.copy()
    
    # Add keys and values from dict2 that are not in dict1
    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = value
    
    return merged_dict

def merge_summaries(summary1: PortersFiveForcesSummary, summary2: PortersFiveForcesSummary) -> PortersFiveForcesSummary:
    '''
    Merge two instances of PortersFiveForcesSummary, keeping unique forces and retaining summary1's version of duplicates.
    '''
    if summary1 is None or summary1 == '':
        summary1 = PortersFiveForcesSummary(forces_scores=[])
    if summary2 is None or summary2 == '':
        summary2 = PortersFiveForcesSummary(forces_scores=[])

    # Create a dictionary for summary1 forces using the original attribute names
    force_dict: Dict[str, PortersForceScore] = {force.force: force for force in summary1.forces}
    
    # Add unique forces from summary2, ignoring duplicates
    for force in summary2.forces:
        if force.force not in force_dict:
            force_dict[force.force] = force
    
    # Create a new summary with merged forces using the alias 'forces_scores'
    merged_summary = PortersFiveForcesSummary(forces_scores=list(force_dict.values()))

    return merged_summary

class State(TypedDict):
    ticker: Annotated[str, lambda a, b: a if len(a)>0 else b] 
    forces: Annotated[Dict, lambda a, b: a if len(a)>0 else b]
    in_spx: Annotated[bool, lambda a, b: a if a else b]
    company_name: Annotated[str, lambda a, b: a if len(a)>0 else b]
    current_premium: Annotated[float, lambda a, b: max(a,b)]
    target_premium: Annotated[float, lambda a, b: a if a != 0 else b] 
    expected_return: Annotated[float, lambda a, b: a if a != 0 else b] 
    force_context: Annotated[Dict, merge_dictionaries]
    #summary: Annotated[str,merge_summaries]
    summary: Annotated[PortersFiveForcesSummary,merge_summaries]
    stock_prices: Annotated[Dict, lambda a, b: a if len(a)>0 else b]
    messages: Annotated[list, operator.add]

def dummy_node(state: State) -> State:
    return {"messages": ["intermediate message"]}

def is_in_snp(ticker: str) -> Tuple[bool, Optional[str]]:
    ''' 
        Checks if the given ticker is a member of the S&P 500 index.
        Disclaimer: uses wikipedia. A production version of this 
        function should use a more reliable source.
    '''
    # Fetch the S&P 500 list from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing the S&P 500 tickers
    table = soup.find('table', {'id': 'constituents'})
    rows = table.find_all('tr')[1:]  # Skip the header row

    # Extract tickers from the table
    sp500_tickers = [row.find_all('td')[0].text.strip() for row in rows]

    # Extract company names from the table
    sp500_companies = [row.find_all('td')[1].text.strip() for row in rows]

    # Check if the given ticker is in the S&P 500 list
    is_in_spx = ticker.upper() in sp500_tickers

    # If the ticker is in the S&P 500 list, print the company name
    if is_in_spx:
        company_name = sp500_companies[sp500_tickers.index(ticker.upper())]
        #print(f"{ticker} is in the S&P 500 index. Company name: {company_name}")
    else:
        company_name = None
        #print(f"{ticker} is not in the S&P 500 index")
    return is_in_spx, company_name

def is_evaluation_valid(state: State) -> bool:
    return True

def populate_company_info(state: State) -> Sequence[str]:
    in_spx, company_name = is_in_snp(state["ticker"])
    state["in_spx"] = in_spx
    state["company_name"] = company_name
    return state

def validate_snp(state: State) -> Sequence[str]:  
    if state["in_spx"]:
        print(f"{state['ticker']} is in the S&P 500 index")
        socketio.emit('progress', {'data': f"{state['ticker']} is in the S&P 500 index"})
        return [
            "Calculate Current Premium", 
            "Estimate Target Premium"
        ]
    else:
        print(f"{state['ticker']} is not in the S&P 500 index")
        socketio.emit('progress', {'data': f"{state['ticker']} is not in the S&P 500 index"})
        return [END]

def get_pe_ratio(ticker: str, useForward = False) -> float:
    '''
        Returns the forward P/E ratio of the given ticker.
    '''
    security = yf.Ticker(ticker)
    metric = 'forwardPE' if useForward else 'trailingPE'
    pe_ratio = security.info[metric]
    return pe_ratio

def get_current_premium(state: State) -> float:
    '''
        Returns the current premium of the given ticker,
        measured in terms of the P/E ratio relative to 
        the average for the S&P 500 index. Note that forward 
        earnings are used for individual securities but trailing 
        earnings are used for the S&P 500 index. This is based 
        on lack of data availability from the source used. 
        A production environment should use consistent multples.
    '''
    pe_ratio_ticker = get_pe_ratio(state['ticker'], useForward = True)
    pe_ratio_snp = get_pe_ratio('SPY')
    current_premium = pe_ratio_ticker / pe_ratio_snp - 1
    return {'current_premium': current_premium}

def estimate_target_premium(state: State) -> State:
    target_premium = float(state['summary'].average_score) * 5/100
    return {'target_premium': target_premium}

def calculate_expected_return(state: State) -> State:
    if state['current_premium'] != 0:
        target_return = state['target_premium'] - state['current_premium']
    else:
        target_return = state['target_premium']
    target_return = min(0.4, target_return)
    target_return = max(-0.4, target_return)
    return {'expected_return': target_return}

cp_builder = StateGraph(State)
cp_builder.add_node("Calculate Current Premium", get_current_premium)

cp_builder.add_edge(START, "Calculate Current Premium")
cp_builder.add_edge("Calculate Current Premium", END)

cp_graph = cp_builder.compile()

forces = {
    "Rivalry": [
        "what are the number and relative size of its competitors?",
        "what is the industry growth rate?",
        "what is the similarity of the company's products to its competitors?",
        "what is the size of exit barriers for competitors?",
        "what is the materiality of fixed costs?"
    ],
    "New Entrants":[
        "what are the economies of scale?",
        "how significant is product differentiation, such as strong brand identity and/or customer loyalty?",
        "what are the capital requirements?",
        "how costly or difficult is it for customers to switch from existing companies to new entrants?",
        "do new entrants have access to distribution channels or are they controlled by existing firms?"
        "do regulations restrict new entrants?"
    ],
    "Suppliers":[
        "how many suppliers are available?",
        "how unique (not easily substituted) are the products or services of suppliers?",
        "how costly is it for the company to switch to other suppliers?",
        "how easy is it for suppliers to integrate forward into the company's industry?",
        "how much do suppliers rely on the company (and industry) for business?"
    ],
    "Customers":[
        "how many customers are there?",
        "how large are the orders of customers?",
        "how unique (not easily substituted) are the companys products or services?",
        "how sensitive are customers to changes in price?",
        "how savvy are customers in negotiating prices?"
    ],
    "Substitutes":[
        "what is the relative price performance of competing products or services?",
        "how willing are customers to switch to substitutes?",
        "are the comapany's products or services differentiated?",
        "are there close substitutes for the company's products or services?",
    ]
}

def set_forces(state: State) -> State:
    state["forces"] = forces
    return state

def get_force_context(state: State, force: str) -> str:
    '''
        Returns the context for the selected Porter's Force.
    '''
    tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    company = state['company_name']
    context = ''
    for sub_force in state['forces'][force]:
        query_string = f'{company}: {sub_force}'
        addl_context = tavily_client.get_search_context(query=query_string)
        context += f'{force}:\n{addl_context}'
    return context

def summzarize_force_context(state: State, force: str, context: str) -> str:
    '''
        Summarize the context for the selected Porter's Force.
    '''
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=4096) 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analysis AI. Your task is to summarize information in a manner that is most relevant for financial analysis."),
        ("user", "Summarize the following text as it relates to {company}'s {force}. While your summary should be based on the information provided, do not tell me that it is. Just provide the summary. \n{context}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "company": state['company_name'],
        "force": force,
        "context": context
                })
    summarized_context = response.content
    return summarized_context

def calculate_force_score(state: State, force: str, context: str) -> PortersForceScore:
    '''
        Uses context to calculate the force score.
    '''
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=4096) 
    prompt_system_text = '''
    You are an expert financial analysis AI.
    Your task is to calculatulate the expected relative outperformance (or underperformance)
    of a given company's stock. To do so, you use porters 5 forces, calculating a score for each 
    force and then combining them to get a final score.
    '''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_system_text),
            ("human", '''
                Populate the pydantic model features for the selected porters 5 force, using the context provided.

                Selected Company: {company}
                Selected Porters 5 Force: {force}
                Context: {context}
            '''),
        ]
    )
    llm_structured_output = llm.with_structured_output(schema=PortersForceScore,include_raw=True) 
    runnable = prompt | llm_structured_output
    response = runnable.invoke({
        "company": state['company_name'],
        "force": force,
        "context": context
                })
    score = response['parsed']
    return score

def calculate_porter_force(state: State, force: str) -> State:
    '''
        Calculates the selected Porter's Force.
    '''
    print(f'Getting context for force: {force}...')
    socketio.emit('progress', {'data': f'Getting context for force: {force}...'})
    context = get_force_context(state, force)
    print(f'Summarizing context for force: {force}...')
    socketio.emit('progress', {'data': f'Summarizing context for force: {force}...'})
    context = summzarize_force_context(state, force, context)
    state['force_context'] = {force: context}
    print(f'Calculating score for force: {force}...')
    socketio.emit('progress', {'data': f'Calculating score for force: {force}...'})
    score = calculate_force_score(state, force, context)
    state['summary'].forces.append(score)
    print(f'Finished processing force: {force}.')
    socketio.emit('progress', {'data': f'Finished processing force: {force}.'})
    return state

def calculate_porter_force_competitive_rivalry(state: State) -> State:
    return calculate_porter_force(state, 'Rivalry')

def calculate_porter_force_new_entrants(state: State) -> State:
    return calculate_porter_force(state, 'New Entrants')

def calculate_porter_force_supplier_power(state: State) -> State:
    return calculate_porter_force(state, 'Suppliers')

def calculate_porter_force_customer_power(state: State) -> State:
    return calculate_porter_force(state, 'Customers')

def calculate_porter_force_substitutes(state: State) -> State:
    return calculate_porter_force(state, 'Substitutes')

def get_stock_data(state: State) -> State:
    ticker = state['ticker']
    expected_return = state['expected_return']
    period = '5y'
    stock_data = yf.download(ticker, period=period)
    closing_prices = stock_data['Close']
    last_close_price = closing_prices.iloc[-1]
    future_dates = pd.date_range(start=closing_prices.index[-1] + timedelta(days=1), periods=365, freq='D')
    future_prices = last_close_price * (1 + expected_return * np.linspace(0, 1, 365))
    future_prices_df = pd.DataFrame({'Close': future_prices}, index=future_dates)
    stock_prices = {
        'labels': closing_prices.index.strftime('%Y-%m-%d').tolist() + future_dates.strftime('%Y-%m-%d').tolist(),
        'datasets': [
            {
                'label': 'History',
                'data': closing_prices.values.tolist() + [None] * 365,
                'borderColor': 'rgba(75, 192, 192, 1)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'fill': False,
            },
            {
                'label': 'Forecast',
                'data': [None] * len(closing_prices) + future_prices.tolist(),
                'borderColor': 'rgba(255, 99, 132, 1)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'fill': False,
            }
        ]
    }
    state['stock_prices'] = stock_prices
    return state

ep_builder = StateGraph(State)

ep_builder.add_node("Estimate Target Premium", estimate_target_premium)

ep_builder.add_node("Rivalry", calculate_porter_force_competitive_rivalry)
ep_builder.add_node("New Entrants", calculate_porter_force_new_entrants)
ep_builder.add_node("Suppliers", calculate_porter_force_supplier_power)
ep_builder.add_node("Customers", calculate_porter_force_customer_power)
ep_builder.add_node("Substitutes", calculate_porter_force_substitutes)

for force_node in forces:
    #ep_builder.add_node(force_node, dummy_node)
    ep_builder.add_edge(START, force_node)
    ep_builder.add_edge(force_node, "Estimate Target Premium")
    #ep_builder.add_conditional_edges(force_node, is_evaluation_valid, {True: "Estimate Target Premium", False: force_node})

ep_builder.add_edge("Estimate Target Premium", END)

ep_graph = ep_builder.compile()

builder = StateGraph(State)
builder.add_node("Lookup Ticker",dummy_node)
builder.add_node("Set Forces", set_forces)
builder.add_node("Populate Company Info", populate_company_info)
builder.add_node("Calculate Current Premium", cp_graph)
builder.add_node("Estimate Target Premium", ep_graph)
builder.add_node("Calculate Expected Return", calculate_expected_return)

builder.add_edge(START, "Lookup Ticker")
builder.add_edge("Lookup Ticker", "Set Forces")
builder.add_edge("Set Forces", "Populate Company Info")

builder.add_conditional_edges("Populate Company Info", validate_snp, [
    "Calculate Current Premium",
    "Estimate Target Premium",
    END
])

builder.add_edge("Estimate Target Premium", "Calculate Expected Return")
builder.add_edge("Calculate Current Premium", "Calculate Expected Return")

builder.add_node("Get Stock Data", get_stock_data)
builder.add_edge("Calculate Expected Return", "Get Stock Data")
builder.add_edge("Get Stock Data", END)

graph = builder.compile()


if __name__ == '__main__':
    app.run(debug=True)

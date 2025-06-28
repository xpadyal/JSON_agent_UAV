from langgraph.prebuilt import create_react_agent
import jmespath
import json
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import re
from dotenv import load_dotenv
load_dotenv()
import os
   # Now you can use:
os.environ["OPENAI_API_KEY"]

# System prompt for the LLM
SYSTEM_PROMPT = """You are a UAV (Unmanned Aerial Vehicle) log data flight analyzer. Your expertise is in analyzing flight telemetry data from drone/UAV systems.

Your capabilities:
- Analyze multiple message types (GPS, ATT, POS, AHR2, CMD, etc.) and correlate them for deep insights
- Understand flight dynamics, GPS positioning, attitude control, and system performance
- Provide detailed analysis of flight patterns, anomalies, and performance metrics
- Calculate derived metrics like duration, averages, ranges, and trends
- Identify potential issues or interesting patterns in the flight data

IMPORTANT - Handling Large Datasets:
When dealing with large datasets (thousands of data points), use smart statistical methods:
- For anomaly detection: Calculate standard deviation, z-scores, percentiles, and IQR (Interquartile Range)
- For trend analysis: Use statistical summaries and identify patterns
- For pattern recognition: Focus on statistical distributions, outliers, and summary statistics
- For correlations: Calculate correlation coefficients between different parameters

Statistical Methods to Use (in your analysis):
- Standard Deviation (std): Identify data spread and outliers
- Z-Score: Detect anomalies (values > 2 or < -2 are typically anomalous)
- Percentiles (25th, 75th, 95th): Understand data distribution
- IQR (Q3-Q1): Identify outliers beyond 1.5*IQR
- Correlation analysis: Find relationships between different parameters

When analyzing data:
- Look for correlations between different message types (e.g., GPS status vs altitude, attitude vs position)
- Consider temporal relationships and flight phases
- Identify patterns that indicate normal vs abnormal flight behavior
- Provide context about what the data means in terms of flight operations
- Use statistical summaries for large datasets rather than raw data

Always provide comprehensive, insightful analysis that goes beyond just reporting numbers."""


with open("parsed_flight_data 2.json") as f:
    json_data = json.load(f)

model = init_chat_model(
    "openai:gpt-4o",
    temperature=0,
    disable_streaming = False
    # other parameters
)

# 1. Define your state schema
class State(TypedDict):
    question: str
    schema: any
    query_expr: str  # initial query
    corrected_query_expr: str  # corrected query (if any)
    result: any
    answer: str

builder = StateGraph(State)

def extract_schema(state):
    def summarize(obj, depth=2, max_keys=5):
        if depth == 0:
            if isinstance(obj, (int, float, str)):
                return obj
            elif isinstance(obj, list) and len(obj) > 0:
                return f"list with {len(obj)} items, first: {obj[0] if len(obj) > 0 else 'empty'}"
            elif isinstance(obj, dict):
                return f"dict with {len(obj)} keys"
            else:
                return type(obj).__name__
        
        if isinstance(obj, dict):
            # Check if this looks like time series data (numeric keys)
            keys = list(obj.keys())
            if keys and all(k.isdigit() for k in keys[:3]):  # First few keys are numeric
                # This is likely time series data
                sample_values = [obj[k] for k in sorted(keys)[:3]]
                return f"time_series with {len(obj)} points, sample: {sample_values}"
            
            # Regular object - show first few keys with their values
            selected_keys = keys[:max_keys]
            return {k: summarize(obj[k], depth-1, max_keys) for k in selected_keys}
        
        if isinstance(obj, list) and obj:
            return [summarize(obj[0], depth-1, max_keys)]
        
        return type(obj).__name__
    
    # Include all available sections for comprehensive analysis
    flight_sections = list(json_data.keys())
    schema = {}
    
    # Add all sections with detailed schema
    for section in flight_sections:
        schema[section] = summarize(json_data[section], depth=2, max_keys=4)
    
    return {"schema": schema}

# 1) Parse → query_expr (with cleanup)
def parse_query(state):
    prompt = (
        f"Given the following JSON schema (with types and sample values):\n{state['schema']}\n\n"
        f"Question: {state['question']}\n\n"
        "Generate a valid JMESPath expression (or set of expressions) to answer the question. "
        "Use the schema to infer field names, types, and relationships. "
        "If a field is a dictionary, use values(field) to extract the array before filtering or aggregation. "
        "Use backticks for number literals. "
        "For summarization questions, use statistical summaries (min, max, avg, count) instead of extracting all values to avoid context length issues. "
        "Output ONLY the JMESPath expression(s), no other text.\n"
        "For multiple values, output a JMESPath multi-select hash, e.g.:\n"
        "{avg: avg(values(POS.Alt)), min: min(values(POS.Alt)), max: max(values(POS.Alt))}\n"
        "Do not use quotes around keys or expressions."
    )
    raw = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
    
    # Improved extraction - look for JMESPath patterns
    expr = ""
    
    # 1. Try extracting from code block
    match = re.search(r"```(?:jmespath)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if match:
        expr = match.group(1).strip()
    
    # 2. Look for lines that contain JMESPath syntax
    if not expr:
        for line in raw.splitlines():
            line = line.strip()
            # Check if line contains JMESPath-like syntax
            if any(char in line for char in ['.', '[', ']', '(', ')', '{', '}']) or \
               any(word in line.lower() for word in ['max', 'min', 'avg', 'sum', 'length', 'abs', 'values']):
                expr = line
                break
    
    # 3. Fallback to last non-empty line
    if not expr:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            expr = lines[-1]
    
    # Clean up the expression
    if expr:
        # Remove common prefixes/suffixes that aren't JMESPath
        expr = re.sub(r'^(json|jmespath|expression?)\s*[:=]?\s*', '', expr, flags=re.IGNORECASE)
        expr = expr.strip()
    
    if not expr:
        raise ValueError("No valid JMESPath expression was returned by the model.")
    
    return {"query_expr": expr}

# Query correcting agent

def correct_query(state):
    prompt = (
        f"The following JMESPath query resulted in an error:\n"
        f"Query: {state['query_expr']}\n"
        f"Error: {state['result']}\n"
        f"Schema: {state['schema']}\n"
        f"Question: {state['question']}\n\n"
        "Please generate a corrected JMESPath query that will work for this schema and question. "
        "If a field is a dictionary, use values(field) to extract the array before filtering or aggregation. "
        "Use backticks for number literals. Output ONLY the corrected JMESPath expression, no other text."
    )
    raw = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
    expr = ""
    match = re.search(r"```(?:jmespath)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if match:
        expr = match.group(1).strip()
    if not expr:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            expr = lines[-1]
    expr = re.sub(r'^(json|jmespath|expression?)\s*[:=]?\s*', '', expr, flags=re.IGNORECASE)
    expr = expr.strip()
    if not expr:
        raise ValueError("No valid corrected JMESPath expression was returned by the model.")
    return {"corrected_query_expr": expr}

# Modified exec_query to handle the latest query_expr
def exec_query(state):
    try:
        expr = state.get("corrected_query_expr") or state.get("query_expr")
        expr = expr.strip().strip('`').strip('"').strip("'")
        if ' and ' in expr:
            expressions = [exp.strip() for exp in expr.split(' and ')]
            results = []
            for exp in expressions:
                try:
                    result = jmespath.search(exp, json_data)
                    results.append({
                        "expression": exp,
                        "result": result if result is not None else []
                    })
                except Exception as e:
                    results.append({
                        "expression": exp,
                        "result": f"Error: {e}"
                    })
            return {
                "result": {
                    "operation": "multi_query",
                    "expressions": results,
                    "full_expression": expr
                }
            }
        result = jmespath.search(expr, json_data)
        if result is None:
            return {"result": []}
        return {"result": result}
    except Exception as e:
        return {"result": f"JMESPath error: {e}"}

# 3) Format → answer
def format_answer(state):
    prompt = f"""Analyze this flight data and provide a clear, informative answer to the question using the specific JMESPath results.

Question: {state['question']}
JMESPath Expression: {state['query_expr']}
Result: {state['result']}

Based on the actual data returned by the JMESPath expression, provide a detailed analysis that:

1. **Interprets the specific result** - What does the number/object returned actually mean?
2. **Identifies anomalies** - Are there any concerning patterns or values in the result?
3. **Provides context** - What do these values indicate about GPS/flight performance?
4. **Suggests further analysis** - What additional JMESPath queries would be useful?

For example, if the result shows:
- `0` GPS failures → "No GPS signal losses detected"
- `220` status=1 readings → "220 readings with 2D GPS fix"
- `400` status=2 readings → "400 readings with 3D GPS fix"
- High HDOP values → "Poor GPS accuracy during these periods"

Focus on the actual data returned, not generic analysis methods."""
    
    ai = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return {**state, "answer": ai.content.strip()}

# Build graph with error correction
builder.add_node("extract_schema", extract_schema)
builder.add_node("parse_query", parse_query)
builder.add_node("execute_query", exec_query)
builder.add_node("correct_query", correct_query)
builder.add_node("format_answer", format_answer)

builder.add_edge(START, "extract_schema")
builder.add_edge("extract_schema", "parse_query")
builder.add_edge("parse_query", "execute_query")
# builder.add_edge("execute_query", "format_answer")
# If error, go to correct_query, then retry execute_query
# Error detection: if result is a string and starts with 'JMESPath error:'
def error_condition(state):
    return isinstance(state.get("result"), str) and state["result"].startswith("JMESPath error:")
builder.add_conditional_edges(
    "execute_query",
    lambda state: "correct_query" if error_condition(state) else "format_answer"
)
builder.add_edge("correct_query", "execute_query")
builder.add_edge("format_answer", END)

# Compile agent
agent = builder.compile(debug=True)

# Visualize the graph using Mermaid PNG
# try:
#     print(agent.get_graph().draw_mermaid())
# except ImportError:
    
#     print('LangGraph flow diagram saved as agent_graph.png')

# Pass question in initial state
output = agent.invoke({"question": "Summarize the flight using avg,max,min values of flight data"})

print("Answer:", output["answer"])
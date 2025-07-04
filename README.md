# JSON_Agent: UAV Flight Data Analyzer

## Overview

**JSON_Agent** is a schema-driven, LLM-powered Python agent for querying and analyzing large UAV (Unmanned Aerial Vehicle) flight log datasets. It leverages JMESPath expressions generated by an LLM to extract relevant data and provide insightful, statistical summaries and anomaly detection from complex flight telemetry logs.

## Features
- **Schema-driven**: Automatically extracts and summarizes the JSON schema for robust, context-aware querying.
- **LLM-powered**: Uses OpenAI GPT-4o (or compatible) to generate JMESPath queries and interpret results.
- **Error correction**: Automatically corrects invalid JMESPath queries and retries.
- **Efficient with large data**: Uses statistical summaries (min, max, avg, count) to avoid context window overflows.
- **Generalizable**: Works with any UAV log structure—no hardcoded field names.
- **Extensible**: Easily add new analysis or data sources.

## Setup

### 1. Clone the repository
```sh
git clone https://github.com/<your-username>/JSON_Agent.git
cd JSON_Agent
```

### 2. Create and activate a virtual environment (recommended)
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 5. Add your flight data
Place your UAV flight log JSON file in the project root (e.g., `parsed_flight_data 2.json`).

## Usage

Run the agent with a question:
```sh
python app.py
```

You can modify the question in `app.py`:
```python
output = agent.invoke({"question": "Summarize the flight using avg,max,min values of flight data"})
print("Answer:", output["answer"])
```

### Example Questions
- "What was the highest altitude?"
- "Find avg, min, max roll angle during the flight."
- "Are there any anomalies in the GPS data?"
- "Summarize the flight using avg, max, min values of flight data."

## Contribution

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

MIT License

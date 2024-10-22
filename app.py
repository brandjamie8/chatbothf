import streamlit as st
import requests
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------- Configuration and Constants ---------------------

# Streamlit App Configuration
st.set_page_config(
    page_title="🏥💬 NHS SUS SQL Query Generator",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Define the NHS SUS APC and OPA table schemas
DATABASE_SCHEMA = """
Table: apc
- UnitID (VARCHAR): Unique identifier for the NHS Trust.
- PatientID (VARCHAR): Unique identifier for the patient.
- EpisodeID (VARCHAR): Unique identifier for the patient episode.
- AdmissionDate (DATE): Date of admission.
- AdmissionType (VARCHAR): Type of admission (e.g., Elective, Emergency).
- MainDiagnosis (VARCHAR): Primary diagnosis code.
- SecondaryDiagnosis (VARCHAR): Secondary diagnosis codes.
- Procedure (VARCHAR): Procedure codes performed during the episode.
- DischargeDate (DATE): Date of discharge.
- LengthOfStay (INT): Total number of days admitted.
- DischargeStatus (VARCHAR): Status at discharge (e.g., Recovered, Transferred).
- Age (INT): Age of the patient at admission.
- Gender (VARCHAR): Gender of the patient.
- Ethnicity (VARCHAR): Ethnic group of the patient.
- DeprivationIndex (INT): Index of Multiple Deprivation score.
- AdmissionSource (VARCHAR): Source of admission (e.g., GP Referral, Accident & Emergency).
- ReadmissionFlag (BOOLEAN): Indicates if the patient was readmitted within 30 days.

Table: opa
- UnitID (VARCHAR): Unique identifier for the NHS Trust.
- PatientID (VARCHAR): Unique identifier for the patient.
- EpisodeID (VARCHAR): Unique identifier for the patient episode.
- AppointmentDate (DATE): Date of the outpatient appointment.
- Department (VARCHAR): Department where the appointment took place (e.g., Cardiology, Orthopedics).
- Procedure (VARCHAR): Procedure codes performed during the appointment.
- Outcome (VARCHAR): Outcome of the appointment (e.g., Attended, Did Not Attend).
- Age (INT): Age of the patient at the time of appointment.
- Gender (VARCHAR): Gender of the patient.
- Ethnicity (VARCHAR): Ethnic group of the patient.
- DeprivationIndex (INT): Index of Multiple Deprivation score.
- ReferralSource (VARCHAR): Source of referral (e.g., GP Referral, Self-Referral).
"""

# --------------------- Helper Functions ---------------------

def extract_sql(response):
    """
    Extracts the SQL query from the model's response using regular expressions.
    """
    if isinstance(response, list):
        response = ''.join(response)  # Concatenate list into string if necessary
    elif not isinstance(response, str):
        response = str(response)  # Convert to string if not already

    # Regular expression to extract SQL queries
    match = re.search(
        r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*?;", 
        response, 
        re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(0)
    else:
        return response  # Return as-is if no SQL keyword is found

def generate_sql_query(prompt_input, api_token, schema):
    """
    Generates an SQL query based on the natural language input using Hugging Face's GPT-J-6B model.
    """
    try:
        # Construct the prompt with instructions and schema
        prompt = (
            "You are an SQL assistant specialized in NHS SUS APC and OPA databases. "
            "Given a natural language query and the database schema below, generate only the corresponding SQL query. "
            "Ensure the query adheres to standard SQL syntax and utilizes the appropriate tables and relationships. "
            "Do not add any explanations or additional text.\n\n"
            f"Database Schema:\n{schema}\n\n"
            "User Query: {prompt_input}\n\nSQL Query:"
        )

        headers = {
            "Authorization": f"Bearer {api_token}"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop_sequences": [";"],
            }
        }

        # API endpoint for GPT-J-6B
        API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 503:
            return "Model is loading. Please try again in a few seconds."
        elif response.status_code != 200:
            return f"Error {response.status_code}: {response.text}"

        generated_text = response.json()[0]['generated_text']
        return generated_text

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "I'm sorry, something went wrong."

# --------------------- Streamlit Application ---------------------

def main():
    # Sidebar Configuration
    with st.sidebar:
        st.title('🏥💬 NHS SUS SQL Query Generator')
        st.write('Convert your natural language queries into SQL statements based on the NHS SUS APC and OPA database schemas.')

        # Hugging Face API Token Input
        api_token = st.text_input(
            'Enter Hugging Face API Token:',
            type='password',
            help='Get your API token from your Hugging Face account settings.'
        )

        # Validate API Token Format (Basic Check)
        if api_token and not api_token.startswith('hf_'):
            st.warning('Please enter a valid Hugging Face API token!', icon='⚠️')
        elif api_token.startswith('hf_'):
            st.success('API Token Verified! You can proceed to enter your query.', icon='✅')

        st.markdown('---')
        st.subheader('Model Parameters')

        # Model Selection (Optional)
        # Currently fixed to GPT-J-6B; can be extended to allow other models
        st.selectbox(
            'Choose a Model',
            ['EleutherAI/gpt-j-6B'],
            disabled=True,  # Disabled since we're only using GPT-J-6B
            help='Currently, only GPT-J-6B is supported.'
        )

        # Temperature Slider
        temperature = st.slider(
            'Temperature',
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help='Controls the randomness of the output. Lower values make the output more deterministic.'
        )

        # Top-p Slider
        top_p = st.slider(
            'Top P',
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help='Controls the diversity of the output. Lower values make the output more focused.'
        )

        # Max Length Slider
        max_length = st.slider(
            'Max Length',
            min_value=50,
            max_value=500,
            value=150,
            step=50,
            help='Maximum number of tokens to generate in the response.'
        )

        st.markdown('---')
        st.markdown(
            '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!'
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Welcome to the NHS SUS SQL Query Generator! Please enter your natural language query to generate an SQL statement."
        }]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.code(message["content"], language='sql')
            else:
                st.write(message["content"])

    # Function to clear chat history
    def clear_chat_history():
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Welcome to the NHS SUS SQL Query Generator! Please enter your natural language query to generate an SQL statement."
        }]

    # Clear Chat History Button
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # User input and response generation
    if prompt := st.chat_input(disabled=not (api_token.startswith('hf_') and len(api_token) > 10)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL query..."):
                # Generate SQL query using Hugging Face Inference API
                response = generate_sql_query(prompt, api_token, DATABASE_SCHEMA)
                sql_query = extract_sql(response)
                placeholder = st.empty()
                placeholder.code(sql_query, language='sql')
        st.session_state.messages.append({"role": "assistant", "content": sql_query})

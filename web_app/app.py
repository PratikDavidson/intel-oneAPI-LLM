import streamlit as st
import requests


API_URLS = {
    "API_URL_QA": "https://api-inference.huggingface.co/models/WaRKiD/bert-large-uncased-whole-word-masking-finetuned-intel-oneapi-llm-dataset",
    "API_URL_TF_QA": "https://api-inference.huggingface.co/models/WaRKiD/distilbert-base-uncased-finetuned-intel-llm-tf-dataset",
    "API_URL_YN_QA": "https://api-inference.huggingface.co/models/WaRKiD/distilbert-base-uncased-finetuned-intel-llm-yn-dataset"
}

headers = {"Authorization": f"Bearer {st.secrets['API_TOKEN']}"}

verbs = set(['am', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'shall', 'will',
            'can', 'should', 'would', 'could', 'must', 'may', 'might', 'do', 'does', 'did'])

st.set_page_config(page_title="Context-Based QA AI", layout="centered")

if 'ans' not in st.session_state:
    st.session_state['ans'] = ''

if 'question_type' not in st.session_state:
    st.session_state['question_type'] = 0


def check_question(question):
    first_word = question.lower().strip().split(' ')[0]
    if first_word in verbs:
        st.session_state['question_type'] = 1
    elif question.rfind('true or false') >= 0:
        st.session_state['question_type'] = 2
    else:
        st.session_state['question_type'] = 0


def arg_max(data):
    if data[0]['score'] > data[1]['score']:
        return data[0]['label']
    return data[1]['label']


@st.cache_resource(show_spinner=False)
def load_models():
    for API_URL in API_URLS:
        requests.post(API_URLS[API_URL], headers=headers)


def query(API_URL, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    while response.status_code != 200:
        response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def execute_query(context, question):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }

    check_question(question)
    answer = query(API_URLS['API_URL_QA'], payload)

    if st.session_state['question_type'] == 1:
        payload = {
            "inputs": question + " " + answer['answer']
        }
        answer = query(
            API_URLS['API_URL_YN_QA'], payload)
        answer = arg_max(answer[0])
        st.session_state['ans'] = answer
    elif st.session_state['question_type'] == 2:
        payload = {
            "inputs": question + " " + answer['answer']
        }
        answer = query(
            API_URLS['API_URL_TF_QA'], payload)
        answer = arg_max(answer[0])
        st.session_state['ans'] = answer
    else:
        st.session_state['ans'] = answer['answer']


header = st.container()
input = st.container()
output = st.container()

with st.spinner('Initiating app...'):
    load_models()

with header:
    st.title('Welcome to context-based QA AI')
    st.header('Description')
    st.write('<div style="text-align:justify">' +
             '''This app is built with BERT models fine-tuned to the "oneAPI Hackathon: The LLM Challenge" dataset.''' + '<div>', unsafe_allow_html=True)

with input:
    st.write('')
    context = st.text_area('Enter context below:')
    question = st.text_input('Enter question below:')
    st.button('Answer', on_click=execute_query,
              args=(context, question))

with output:
    st.write('')
    st.code("Answer: " + st.session_state['ans'])

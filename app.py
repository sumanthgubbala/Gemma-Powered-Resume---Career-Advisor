import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
from vector_db import load_and_add_pdf,get_retriever,vectore_reset

# Initialize the model and prompt template
model = OllamaLLM(model="gemma3:1b")

template = """
You are an expert career advisor specializing in resume building and optimization.

Based on the following information, provide clear, practical suggestions and answer the question effectively.

Additionally, share your own professional opinion or recommendation on the matter to add extra value.

Relevant Information:
{data}

Question:
{question}

Respond in a professional, helpful, insightful, and concise manner.

Finally, provide a Resume Rating out of 10 based on relevance, clarity, structure, achievements, and overall presentation.
"""



prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit app layout
st.title("Resume Advisor AI")
st.markdown("Ask questions about resume building and get expert advice powered by AI.")



# Define PDF folder path
PDF_FOLDER = "./pdf"

# Ensure PDF folder exists
os.makedirs(PDF_FOLDER, exist_ok=True)

# Initialize session state for file uploader key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
retriever = None  # Initialize retriever

# PDF upload section
st.subheader("Upload Resume (PDF)")
uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf",key=f"pdf_uploader_{st.session_state.uploader_key}")

if uploaded_file:
    pdf_path = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name} to {PDF_FOLDER}")

    # Dynamically pass the uploaded path to vector.py function
    num_docs = load_and_add_pdf(pdf_path)
    st.success(f"Added {num_docs} pages to vector database.")
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.retriever = get_retriever()


# Use existing retriever if already available
retriever = st.session_state.get("retriever", None)

# Input form for question
with st.form(key="question_form"):

    question = st.text_area("Enter your question (type 'q' to quit):", placeholder="e.g., How do I highlight my skills on a resume?")
    
    submit_button = st.form_submit_button("Get Advice")


# Handle form submission
if submit_button and question:
    if question.lower() == 'q':
        st.warning("Application stopped.")
        st.stop()
    else:
        with st.spinner("Generating advice..."):
            try:
                # Invoke retriever and chain
                data = retriever.invoke(question)
                if not data:
                    st.error("No data retrieved from the resume. Please check your PDF.")
                else:
                    result = chain.invoke({
                        "data": data,
                        "question": question
                    })
                # Display result
                st.subheader("Advice:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Quit button
if st.button("Quit"):
    for file in os.listdir(PDF_FOLDER):
        file_path = os.path.join(PDF_FOLDER, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # Increment uploader key to force file uploader reset
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1

    # Clear Streamlit cache to ensure no residual widget state
    st.cache_data.clear()
    st.cache_resource.clear()
    vectore_reset()

    # Clear session variables
    st.session_state.pop("uploaded_file_name", None)
    st.session_state.pop("retriever", None)

    st.success("All data cleared. Restarting app...")
    st.rerun()

# Add styling
st.markdown("""
<style>
    .stTextInput > div > input {
        border: 2px solid #4B5EAA;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4B5EAA;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background-color: #3A4A88;
    }
    .stFileUploader > div > button {
        background-color: #4B5EAA;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

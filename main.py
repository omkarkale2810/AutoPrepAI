import streamlit as st
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import RestrictedPython
from RestrictedPython import compile_restricted, safe_globals, limited_builtins
import io
import sys
import uuid
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import traceback
import time


st.set_page_config(
    page_title="Data Preprocessor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    try:
        llm = ChatGroq(
            model="llama3-70b-8192", 
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

llm, embeddings = initialize_models()

st.sidebar.title("Configuration")
preprocessing_mode = st.sidebar.selectbox(
    "Preprocessing Mode",
    ["Natural Language", "Template-based"]
)

with st.sidebar.expander("Advanced Options"):
    max_code_length = st.slider("Max Code Length", 500, 2000, 1000)
    enable_visualization = st.checkbox("Enable Visualizations", True)
    save_history = st.checkbox("Save Processing History", True)
    chunk_size = st.slider("Document Chunk Size", 300, 800, 500)

st.markdown('<h1 class="main-header">LLM-Powered Data Preprocessing</h1>', unsafe_allow_html=True)
st.markdown("Upload a dataset and preprocess it using natural language instructions powered by AI.")


session_vars = [
    "df", "preprocessed_df", "summary", "processing_history", 
    "data_quality_report", "preprocessing_steps", "code_generated",
    "conversation_active", "conversation_history", "current_step", 
    "auto_mode", "preprocessing_complete", "suggested_steps", "step_counter"
]

for var in session_vars:
    if var not in st.session_state:
        if var in ["processing_history", "conversation_history"]:
            st.session_state[var] = []
        elif var == "step_counter":
            st.session_state[var] = 0
        else:
            st.session_state[var] = None if var != "processing_history" else []

def analyze_data_quality(df):
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_data": {},
        "duplicates": df.duplicated().sum(),
        "data_types": {},
        "numeric_columns": [],
        "categorical_columns": [],
        "outliers": {},
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2 
    }
    for col in df.columns:
        missing_count = df[col].isna().sum()
        quality_report["missing_data"][col] = {
            "count": int(missing_count),
            "percentage": round((missing_count / len(df)) * 100, 2)
        }
        quality_report["data_types"][col] = str(df[col].dtype)
        
        if df[col].dtype in ['int64', 'float64']:
            quality_report["numeric_columns"].append(col)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            quality_report["outliers"][col] = int(outlier_count)
        else:
            quality_report["categorical_columns"].append(col)
    
    return quality_report

def visualize_data_quality(df, quality_report):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Data Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=True, ax=ax, cmap='viridis')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Data Types Distribution")
        type_counts = pd.Series(list(quality_report["data_types"].values())).value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, title="Data Types")
        st.plotly_chart(fig, use_container_width=True)
    
    if quality_report["missing_data"]:
        missing_df = pd.DataFrame([
            {"Column": col, "Missing %": info["percentage"]} 
            for col, info in quality_report["missing_data"].items()
            if info["percentage"] > 0
        ])
        
        if not missing_df.empty:
            st.subheader("Missing Data by Column")
            fig = px.bar(missing_df, x="Column", y="Missing %", title="Missing Data Percentage")
            st.plotly_chart(fig, use_container_width=True)

def create_enhanced_vector_store(df, prompt):
    schema_info = []
    for col in df.columns:
        col_info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique()),
            "missing_percentage": round((df[col].isna().sum() / len(df)) * 100, 2)
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None
            })
        else:
            top_values = df[col].value_counts().head(3)
            col_info["top_values"] = top_values.to_dict()
        
        schema_info.append(json.dumps(col_info, indent=2))
    
    quality_report = analyze_data_quality(df)
    quality_text = f"""
    Dataset Quality Summary:
    - Total rows: {quality_report['total_rows']}
    - Total columns: {quality_report['total_columns']}
    - Duplicate rows: {quality_report['duplicates']}
    - Memory usage: {quality_report['memory_usage']:.2f} MB
    - Numeric columns: {', '.join(quality_report['numeric_columns'])}
    - Categorical columns: {', '.join(quality_report['categorical_columns'])}
    """
    
    # combine all information to store into vector database 
    documents_text = [
        "Dataset Schema Information:",
        *schema_info,
        quality_text,
        f"User Request: {prompt}"
    ]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    documents = []
    
    for i, text in enumerate(documents_text):
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"source": f"dataset_info_{i}", "type": "schema"}
            ))
    
    return FAISS.from_documents(documents, embeddings)

  
def safe_execute_code(code, df):
    from RestrictedPython.PrintCollector import PrintCollector
    
    # restricted python environment to run the code 
    restricted_globals = safe_globals.copy()
    restricted_globals.update({
        "pd": pd,
        "np": np,
        "LabelEncoder": LabelEncoder,
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
        "OneHotEncoder": OneHotEncoder,
        "OrdinalEncoder": OrdinalEncoder,
        "SimpleImputer": SimpleImputer,
        "KNNImputer": KNNImputer,
        "SelectKBest": SelectKBest,
        "f_classif": f_classif,
        "f_regression": f_regression,
        "df": df.copy(),
        "_print_": PrintCollector,
        "_getattr_": getattr,
        "_getitem_": lambda obj, index: obj[index],
        "_getiter_": iter,
        "_iter_unpack_sequence_": lambda it, spec: list(it),
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
    })
    restricted_globals["__builtins__"] = limited_builtins
    
    try:
        
        compiled_code = compile_restricted(code, "<string>", "exec")
        if compiled_code is None:
            raise ValueError("Code compilation failed - potentially unsafe code detected")
        
        exec(compiled_code, restricted_globals)
        
        result_df = restricted_globals.get("df")
        
        if result_df is None:
            raise ValueError("Code did not return a DataFrame")
        
        print_output = ""
        if "_print" in restricted_globals:
            print_collector = restricted_globals["_print"]
            if hasattr(print_collector, 'txt'):
                print_output = print_collector.txt
        
        return result_df, print_output
        
    except Exception as e:
        return None, f"Execution Error: {str(e)}"


# --------------------------------------------------------------------------------
def analyze_data_and_suggest_steps(df):
    
    quality_report = analyze_data_quality(df)
    prompt = f"""
    You are an expert data scientist. Analyze this dataset and provide a comprehensive preprocessing plan.
    
    DATASET ANALYSIS:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Missing data: {dict(quality_report['missing_data'])}
    - Duplicates: {quality_report['duplicates']}
    - Numeric columns: {quality_report['numeric_columns']}
    - Categorical columns: {quality_report['categorical_columns']}
    - Outliers: {dict(quality_report['outliers'])}
    
    Provide a step-by-step preprocessing plan as a data scientist would approach this dataset.
    Format your response as numbered steps, each step should be a single actionable preprocessing task.
    
    Example format:
    1. Remove duplicate rows (found X duplicates)
    2. Handle missing values in column Y using median imputation
    3. Encode categorical variable Z using label encoding
    4. Scale numerical features using StandardScaler
    5. Remove outliers from column A using IQR method
    
    Provide 5-10 specific steps based on the actual data issues found.
    """
    
    try:
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            steps_text = response.content.strip()
        else:
            steps_text = str(response).strip()
        
        steps = []
        lines = steps_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
    
                step = line.split('.', 1)[-1].strip() if '.' in line else line.strip('- ')
                if step:
                    steps.append(step)
        
        return steps[:12] 
        
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return [
            "Remove duplicate rows if any",
            "Handle missing values using appropriate imputation",
            "Encode categorical variables",
            "Scale numerical features",
            "Handle outliers if present"
        ]

def generate_step_code(step_description, df, conversation_context=""):

    data_info = f"""
    Current dataset info:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Data types: {dict(df.dtypes.astype(str))}
    - Missing values: {dict(df.isnull().sum())}
    - Sample data: {df.head(2).to_dict()}
    """
    
    prompt = f"""
    You are an expert data scientist. Generate Python code for this specific preprocessing step.
    
    STEP TO IMPLEMENT: {step_description}
    
    CURRENT DATASET INFO:
    {data_info}
    
    CONVERSATION CONTEXT:
    {conversation_context}
    
    REQUIREMENTS:
    1. Work with DataFrame variable 'df'
    2. Implement ONLY the specified step
    3. Add informative print statements about what's being done
    4. Use try-except for error handling
    5. Return the modified DataFrame at the end
    6. Be specific and targeted - don't do extra steps
    
    EXAMPLE STRUCTURE:
    ```python
    print("Step X: [Description of what you're doing]")
    try:
        # Your specific preprocessing code here
        print(f"Action completed successfully. New shape: {{df.shape}}")
    except Exception as e:
        print(f"Error in step: {{e}}")
    
    df
    ```
    
    Generate ONLY the Python code, no explanations outside the code.
    """
    
    try:
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            code = response.content.strip()
        else:
            code = str(response).strip()
        
        # Clean code
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
            
        return code.strip()
        
    except Exception as e:
        return f"""
print("Error generating code: {str(e)}")
print("Skipping this step...")
df
"""
# ------------------------------------------------------------------------------------


def add_to_conversation(role, message, code=None, output=None):
    entry = {
        "role": role,
        "message": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "step": st.session_state.step_counter
    }
    if code:
        entry["code"] = code
    if output:
        entry["output"] = output
    
    st.session_state.conversation_history.append(entry)

def display_conversation():
    st.subheader("Conversation")
    for entry in st.session_state.conversation_history:
        if entry["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(f"**[{entry['timestamp']}]** {entry['message']}")
                if "code" in entry:
                    st.code(entry["code"], language="python")
                if "output" in entry:
                    st.text(entry["output"])
        else:
            with st.chat_message("user"):
                st.write(f"**[{entry['timestamp']}]** {entry['message']}")

def execute_preprocessing_step(step_description, df):
    st.session_state.step_counter += 1
    
    context = "\n".join([f"{entry['role']}: {entry['message']}" 
                        for entry in st.session_state.conversation_history[-3:]])
    
    code = generate_step_code(step_description, df, context)
    
    result_df, output = safe_execute_code(code, df)
    
    add_to_conversation("assistant", f"Executing step {st.session_state.step_counter}: {step_description}", code, output)
    
    if result_df is not None:
        return result_df, output, True
    else:
        return df, output, False

st.header("Upload Your Dataset in CSV format")
uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    help="Upload your dataset in CSV format"
)

if uploaded_file:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        
        #display basic information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(st.session_state.df))
        with col2:
            st.metric("Columns", len(st.session_state.df.columns))
        with col3:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", st.session_state.df.duplicated().sum())
        
        st.session_state.data_quality_report = analyze_data_quality(st.session_state.df)
        
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

        if enable_visualization:
            st.subheader("Data Quality Analysis")
            visualize_data_quality(st.session_state.df, st.session_state.data_quality_report)
            
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")


if st.session_state.df is not None:
    st.header("Conversational Data Preprocessing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Auto Mode (AI-Driven)", type="primary"):
            st.session_state.auto_mode = True
            st.session_state.conversation_active = True
            st.session_state.suggested_steps = analyze_data_and_suggest_steps(st.session_state.df)
            add_to_conversation("user", "Please preprocess this dataset automatically using your expertise")
            add_to_conversation("assistant", f"I'll analyze your dataset and perform {len(st.session_state.suggested_steps)} preprocessing steps:")
            for i, step in enumerate(st.session_state.suggested_steps, 1):
                add_to_conversation("assistant", f"Step {i}: {step}")
    
    with col2:
        if st.button("👤 Manual Mode (Step-by-step)", type="secondary"):
            st.session_state.auto_mode = False
            st.session_state.conversation_active = True
            st.session_state.suggested_steps = analyze_data_and_suggest_steps(st.session_state.df)
            add_to_conversation("user", "I want to preprocess this dataset step by step")
            add_to_conversation("assistant", "Great! I've analyzed your dataset. Here are my suggestions:")
            for i, step in enumerate(st.session_state.suggested_steps, 1):
                add_to_conversation("assistant", f"Suggestion {i}: {step}")
            add_to_conversation("assistant", "You can choose which steps to execute or give me custom instructions.")
    
    with col3:
        if st.button("🔄 Reset Conversation"):
            for var in session_vars:
                if var == "conversation_history":
                    st.session_state[var] = []
                elif var == "step_counter":
                    st.session_state[var] = 0
                else:
                    st.session_state[var] = False
            st.rerun()
    
    # display conversation
    if st.session_state.conversation_active:
        display_conversation()
        
        # auto mode execution
        if st.session_state.auto_mode and st.session_state.suggested_steps and not st.session_state.preprocessing_complete:
            st.subheader("Auto Preprocessing in Progress...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            current_df = st.session_state.df.copy()
            
            for i, step in enumerate(st.session_state.suggested_steps):
                status_text.text(f"Executing step {i+1}/{len(st.session_state.suggested_steps)}: {step}")
                progress_bar.progress((i) / len(st.session_state.suggested_steps))
                
                # Execute step
                result_df, output, success = execute_preprocessing_step(step, current_df)
                
                if success:
                    current_df = result_df
                    st.success(f"Step {i+1} completed successfully!")
                else:
                    st.error(f"Step {i+1} failed: {output}")
                
                time.sleep(0.1)
            
            progress_bar.progress(1.0)
            status_text.text("Auto preprocessing completed!")
            
            st.session_state.preprocessed_df = current_df
            st.session_state.preprocessing_complete = True
            
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Dataset:**")
                st.write(f"Shape: {st.session_state.df.shape}")
                st.write(f"Missing values: {st.session_state.df.isnull().sum().sum()}")
                st.dataframe(st.session_state.df.head(3))
            
            with col2:
                st.write("**Preprocessed Dataset:**")
                st.write(f"Shape: {current_df.shape}")
                st.write(f"Missing values: {current_df.isnull().sum().sum()}")
                st.dataframe(current_df.head(3))
            
            add_to_conversation("assistant", "Auto preprocessing completed! Your dataset is now ready for analysis.")
        
        # Manual mode - user input
        elif not st.session_state.auto_mode and not st.session_state.preprocessing_complete:
            st.subheader("Your Turn!")

            if st.session_state.suggested_steps:
                st.write("**Quick Actions (Click to execute):**")
                cols = st.columns(min(3, len(st.session_state.suggested_steps)))
                
                for i, step in enumerate(st.session_state.suggested_steps[:3]):
                    with cols[i]:
                        if st.button(f"{step[:30]}...", key=f"quick_{i}"):
                            current_df = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.df
                            
                            add_to_conversation("user", f"Execute: {step}")
                            result_df, output, success = execute_preprocessing_step(step, current_df)
                            
                            if success:
                                st.session_state.preprocessed_df = result_df
                                st.success("Step executed successfully!")
                                st.rerun()
                            else:
                                st.error(f"Execution failed: {output}")
            
            # typed instruction input
            st.write("**Or give custom instructions:**")
            user_input = st.text_input(
                "Enter your preprocessing instruction:",
                placeholder="e.g., 'remove rows where age is null', 'encode the category column', 'scale all numeric features'",
                key="user_instruction"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Execute Instruction") and user_input:
                    current_df = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.df
                    
                    add_to_conversation("user", user_input)
                    result_df, output, success = execute_preprocessing_step(user_input, current_df)
                    
                    if success:
                        st.session_state.preprocessed_df = result_df
                        st.success("Instruction executed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Execution failed!!!: {output}")
            
            with col2:
                if st.button("Finish Preprocessing"):
                    st.session_state.preprocessing_complete = True
                    final_df = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.df
                    add_to_conversation("assistant", "Preprocessing session completed! Your dataset is ready.")
                    st.success("Preprocessing completed!")
                    st.rerun()
        
        if st.session_state.conversation_active:
            with st.expander("Current Dataset Status", expanded=False):
                current_df = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.df
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Rows", len(current_df))
                with col2:
                    st.metric("Current Columns", len(current_df.columns))
                with col3:
                    st.metric("Missing Values", current_df.isnull().sum().sum())
                with col4:
                    st.metric("Steps Completed", st.session_state.step_counter)
                
                st.dataframe(current_df.head(5), use_container_width=True)
        
        #download option
        if st.session_state.preprocessing_complete and st.session_state.preprocessed_df is not None:
            st.subheader("Download Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = st.session_state.preprocessed_df.to_csv(index=False)
                st.download_button(
                    "Download Preprocessed CSV",
                    data=csv_data,
                    file_name=f"preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                conversation_report = "# Preprocessing Conversation Report\n\n"
                for entry in st.session_state.conversation_history:
                    conversation_report += f"**{entry['role'].title()}** [{entry['timestamp']}]:\n{entry['message']}\n\n"
                    if "code" in entry:
                        conversation_report += f"```python\n{entry['code']}\n```\n\n"
                    if "output" in entry:
                        conversation_report += f"Output:\n```\n{entry['output']}\n```\n\n"
                    conversation_report += "---\n\n"
                
                st.download_button(
                    "Download Conversation Report",
                    data=conversation_report,
                    file_name=f"preprocessing_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col3:
                # Generate summary code
                all_codes = [entry.get('code', '') for entry in st.session_state.conversation_history if entry.get('code')]
                full_code = f"# Generated Preprocessing Pipeline\n# Generated on {datetime.now()}\n\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import *\nfrom sklearn.impute import *\n\n# Load your dataset\n# df = pd.read_csv('your_dataset.csv')\n\n" + "\n\n".join(all_codes)
                
                st.download_button(
                    "💻 Download Complete Code",
                    data=full_code,
                    file_name=f"preprocessing_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                    mime="text/plain"
                )
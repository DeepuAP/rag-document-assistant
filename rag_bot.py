import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import openai # specific library for ChatGPT and DeepSeek

# --- 1. CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="NeuraDoc AI", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. SETUP & STATE ---
load_dotenv()

# Initialize Session State for Keys and Settings
if "settings" not in st.session_state:
    st.session_state.settings = {
        "provider": "Google Gemini",
        "model": "gemini-1.5-flash",
        "api_keys": {
            "Google Gemini": os.getenv("GOOGLE_API_KEY") or "",
            "OpenAI (ChatGPT)": "",
            "DeepSeek": ""
        }
    }

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- 3. CUSTOM CSS (Professional Theme) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header & Nav Bar */
    .nav-container {
        background: white;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat Bubbles */
    .user-msg {
        background-color: #1e40af;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 12px 12px 2px 12px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.5;
    }
    
    .ai-msg {
        background-color: white;
        color: #334155;
        padding: 1rem 1.25rem;
        border-radius: 12px 12px 12px 2px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 85%;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        line-height: 1.5;
    }
    
    /* Upload Box */
    .upload-box {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 800px;
    }
    
    /* Chat Input Fixed */
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        z-index: 1000;
    }
    
    /* Main content padding for fixed footer */
    .main-block {
        padding-bottom: 150px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. LOGIC FUNCTIONS (Multi-Model Support) ---

@st.dialog("⚙️ AI Configuration Settings")
def open_settings():
    st.write("Configure your AI providers and keys here.")
    
    # Provider Selection
    provider = st.selectbox(
        "Select AI Provider", 
        ["Google Gemini", "OpenAI (ChatGPT)", "DeepSeek"],
        index=["Google Gemini", "OpenAI (ChatGPT)", "DeepSeek"].index(st.session_state.settings["provider"])
    )
    
    # Dynamic Model Selection based on Provider
    model_options = []
    if provider == "Google Gemini":
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    elif provider == "OpenAI (ChatGPT)":
        model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif provider == "DeepSeek":
        model_options = ["deepseek-chat", "deepseek-coder"]
        
    current_model = st.session_state.settings["model"]
    if current_model not in model_options:
        current_model = model_options[0]
        
    model = st.selectbox("Select Model", model_options, index=model_options.index(current_model))
    
    # API Key Input
    current_key = st.session_state.settings["api_keys"][provider]
    new_key = st.text_input(f"{provider} API Key", value=current_key, type="password", help=f"Enter your key for {provider}")
    
    if st.button("💾 Save Settings", type="primary"):
        st.session_state.settings["provider"] = provider
        st.session_state.settings["model"] = model
        st.session_state.settings["api_keys"][provider] = new_key
        st.rerun()

def get_ai_response(context_text, user_question):
    settings = st.session_state.settings
    provider = settings["provider"]
    api_key = settings["api_keys"][provider]
    model_name = settings["model"]
    
    if not api_key:
        return f"⚠️ Error: Please enter an API Key for {provider} in Settings."

    system_prompt = f"""
    You are a professional Corporate Document Assistant.
    Answer the user's question based ONLY on the provided context.
    Keep the tone professional, precise, and helpful.
    
    CONTEXT:
    {context_text}
    """

    try:
        # --- GOOGLE GEMINI ---
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            # Gemini combines system/user prompts slightly differently, but this works well
            full_prompt = f"{system_prompt}\n\nQUESTION:\n{user_question}"
            response = model.generate_content(full_prompt)
            return response.text

        # --- OPENAI (ChatGPT) ---
        elif provider == "OpenAI (ChatGPT)":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ]
            )
            return response.choices[0].message.content

        # --- DEEPSEEK (OpenAI Compatible) ---
        elif provider == "DeepSeek":
            client = openai.OpenAI(
                api_key=api_key, 
                base_url="https://api.deepseek.com"
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                stream=False
            )
            return response.choices[0].message.content
            
    except Exception as e:
        return f"❌ Error with {provider}: {str(e)}"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception as e:
            continue
    return text

# --- 5. UI LAYOUT ---

# Top Nav Bar (Simulated with Columns)
col_nav1, col_nav2 = st.columns([4, 1])

with col_nav1:
    st.markdown("""
    <div style="padding-top: 10px;">
        <h1 style="color: #0f172a; font-size: 1.8rem; font-weight: 800; margin: 0; display: flex; align-items: center; gap: 0.75rem;">
            <span style="color: #2563eb;">⚡</span> NeuraDoc AI
        </h1>
        <p style="color: #64748b; font-size: 0.9rem; margin: 0;">Enterprise Document Intelligence • <span style="color:#2563eb; font-weight:600;">{provider} ({model})</span></p>
    </div>
    """.format(provider=st.session_state.settings["provider"], model=st.session_state.settings["model"]), unsafe_allow_html=True)

with col_nav2:
    if st.button("⚙️ Settings", use_container_width=True):
        open_settings()

st.markdown("---")

# Main Content Area
main_container = st.container()

with main_container:
    # Use a div to pad the bottom so fixed chat doesn't cover content
    st.markdown('<div class="main-block">', unsafe_allow_html=True)

    # --- SCENARIO 1: NO DOCUMENTS UPLOADED ---
    if not st.session_state.pdf_text:
        st.markdown("""
        <div class="upload-box">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <h2 style="color: #1e293b; font-weight: 700;">Upload Knowledge Base</h2>
            <p style="color: #64748b; margin-bottom: 2rem;">Upload PDF documents to begin AI analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the uploader
        col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
        with col_up2:
            uploaded_files = st.file_uploader(" ", accept_multiple_files=True, type=['pdf'], label_visibility="collapsed")
            
            if uploaded_files:
                if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                    with st.spinner("Processing content..."):
                        raw_text = get_pdf_text(uploaded_files)
                        if raw_text:
                            st.session_state.pdf_text = raw_text
                            st.session_state.uploaded_files = uploaded_files
                            st.rerun()
                        else:
                            st.error("Could not extract text from files.")

    # --- SCENARIO 2: DOCUMENTS LOADED (DASHBOARD MODE) ---
    else:
        col1, col2 = st.columns([2.5, 1], gap="large")
        
        # Left Column: Chat History
        with col1:
            st.markdown('<div style="margin-bottom: 1rem; font-weight: 600; color: #334155;">💬 Conversation History</div>', unsafe_allow_html=True)
            
            # Welcome Message
            if not st.session_state.chat_history:
                 st.markdown(f"""
                <div class="ai-msg">
                    Hello! I'm ready to answer questions about your <b>{len(st.session_state.uploaded_files)} document(s)</b> 
                    using <b>{st.session_state.settings['provider']}</b>.
                </div>
                """, unsafe_allow_html=True)
            
            # Render History
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f'<div class="user-msg">{chat["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-msg">{chat["message"]}</div>', unsafe_allow_html=True)

        # Right Column: Dashboard
        with col2:
            st.markdown('<div style="margin-bottom: 1rem; font-weight: 600; color: #334155;">📊 Insights & Tools</div>', unsafe_allow_html=True)
            
            # Stat Card
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 2.5rem; font-weight: 700; color: #2563eb;">{len(st.session_state.uploaded_files)}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight: 500;">Active Documents</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='color: #1e293b; margin-top: 1.5rem;'>⚡ Quick Actions</h3>", unsafe_allow_html=True)
            
            if st.button("📋 Summarize All", use_container_width=True):
                with st.spinner("Generating summary..."):
                    response = get_ai_response(st.session_state.pdf_text, "Summarize the key information and main topics.")
                    st.session_state.chat_history.append({"role": "assistant", "message": f"**Summary:**\n\n{response}"})
                    st.rerun()
            
            if st.button("🗝️ Extract Key Points", use_container_width=True):
                with st.spinner("Extracting..."):
                    response = get_ai_response(st.session_state.pdf_text, "List the most critical points, dates, and figures in bullet format.")
                    st.session_state.chat_history.append({"role": "assistant", "message": f"**Key Points:**\n\n{response}"})
                    st.rerun()
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. CHAT INPUT (FIXED AT BOTTOM) ---
if st.session_state.pdf_text:
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        
        # Check if key is missing before trying
        current_provider = st.session_state.settings["provider"]
        if not st.session_state.settings["api_keys"][current_provider]:
             st.session_state.chat_history.append({"role": "assistant", "message": f"⚠️ **Configuration Error:** Please click 'Settings' and enter an API Key for {current_provider}."})
             st.rerun()
        else:
            try:
                with st.spinner(f"Thinking with {current_provider}..."):
                    response = get_ai_response(st.session_state.pdf_text, prompt)
                    st.session_state.chat_history.append({"role": "assistant", "message": response})
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")
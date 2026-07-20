import streamlit as st
import fitz  # PyMuPDF
from google import genai
from groq import Groq
from openai import OpenAI
import docx2txt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy
import re

# 1. Page Configuration
st.set_page_config(page_title="Circuit Study Buddy", layout="centered")
st.title("🔌 Circuit Study Buddy")
st.markdown("Upload notes or take a live photo. Images are processed using advanced Multimodal AI Vision.")

# 2. Get API Keys securely
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    st.error("❌ Missing API keys in `.streamlit/secrets.toml`!")
    st.stop()

# 3. Smart Parsing (With Progress Bar & Speed Limits)
def process_input_source(file_obj):
    extracted_text = ""
    file_type = ""
    
    if hasattr(file_obj, 'name'):
         file_type = file_obj.name.split('.')[-1].lower()
         
    try:
        if file_type == 'pdf':
            doc = fitz.open(stream=file_obj.read(), filetype="pdf")
            total_pages = len(doc)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_pages = min(total_pages, 30) 
            if total_pages > 30:
                st.warning(f"⚠️ Document is {total_pages} pages long. To maintain high speed, only the first 30 pages are being loaded.")
                
            for page_num in range(max_pages):
                page = doc.load_page(page_num)
                extracted_text += f"\n\n--- PAGE {page_num + 1} ---\n\n{page.get_text('text')}"
                progress_bar.progress((page_num + 1) / max_pages)
                status_text.text(f"Reading page {page_num + 1} of {max_pages}...")
                
            progress_bar.empty()
            status_text.empty()
                
        elif file_type == 'txt':
            extracted_text = file_obj.getvalue().decode("utf-8")
            
        elif file_type in ['docx', 'doc']:
            extracted_text = docx2txt.process(file_obj)
            
        else:
            st.session_state.source_image = Image.open(file_obj)
            extracted_text = "📸 Image successfully loaded into the visual memory buffer."
            
    except Exception as e:
        extracted_text = f"Error processing file: {e}"
        
    return extracted_text

# 4. Multi-Model Router Function
def get_ai_response(prompt, system_context, gemini_key, groq_key, or_key, image=None):
    try:
        client = genai.Client(api_key=gemini_key)
        if image:
            contents = [f"System Context:\n{system_context}\n\nUser Question: {prompt}", image]
        else:
            contents = f"System Context:\n{system_context}\n\nUser Question: {prompt}"
            
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=contents
        )
        return response.text, "Gemini 3.5 Flash"
        
    except Exception as gemini_error:
        st.warning(f"⚠️ Gemini Error: {gemini_error}")
        try:
            groq_client = Groq(api_key=groq_key)
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192", 
                messages=[{"role": "system", "content": system_context}, {"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content, "Groq (Llama 3)"
        except Exception as groq_error:
            st.warning(f"⚠️ Groq Error: {groq_error}")
            try:
                or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)
                completion = or_client.chat.completions.create(
                    model="openai/gpt-oss-20b:free",
                    messages=[{"role": "system", "content": system_context}, {"role": "user", "content": prompt}]
                )
                return completion.choices[0].message.content, "OpenRouter"
            except Exception as or_error:
                st.error(f"❌ OpenRouter Error: {or_error}")
                return "All networks failed. Read the specific error codes above to debug.", "Error"

# 5. Initialize Persistent States
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "source_image" not in st.session_state:
    st.session_state.source_image = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Sidebar Controls for Document & Camera
with st.sidebar:
    st.header("Document Source")
    
    source_choice = st.radio("Choose input method:", ["Upload File", "Live Camera"])
    uploaded_file = None
    camera_photo = None
    
    if source_choice == "Upload File":
        uploaded_file = st.file_uploader("Upload your notes", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
    else:
        camera_photo = st.camera_input("Take a picture of your circuit diagram or equations")

    active_source = uploaded_file if source_choice == "Upload File" else camera_photo
    
    if active_source and st.session_state.document_text is None:
        with st.spinner("Processing source..."):
            st.session_state.document_text = process_input_source(active_source)
            st.success("Source loaded successfully!")
            
    if st.button("Clear Data & Chat History"):
        st.session_state.messages = []
        st.session_state.document_text = None
        st.session_state.source_image = None
        st.rerun()

# 7. Stream Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "model" in msg:
            st.caption(f"⚡ Answered by: {msg['model']}")

# 8. User Input & AI Execution
if prompt := st.chat_input("Ask a question about your notes..."):
    
    if not st.session_state.document_text and st.session_state.source_image is None:
        st.error("Please upload a file or take a photo before asking a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    system_instruction = (
        "You are an advanced engineering tutor specializing in mathematics and computing. "
        "Systematically analyze the provided document text or image to answer the user's questions.\n\n"
        "CRITICAL FORMATTING RULE: You MUST format ALL mathematical equations, variables, matrices, and fractions using strict LaTeX. "
        "Enclose inline math with single $and display math with double$$. \n"
        "LATEX SYNTAX WARNING: You must meticulously check that every opening $or$$ has a matching closing tag. "
        "If you use a LaTeX environment (like cases, matrix, or aligned), you MUST include both the \\begin{} and \\end{} tags. Do not output broken LaTeX.\n\n"
        "Format your response with:\n"
        "1. **Step-by-Step Solution:** Point out if stated components differ from standard assumptions (e.g., current vs. voltage sources).\n"
        "2. **💡 Short Trick:** Provide a calculation shortcut or verification method.\n"
        "3. **📊 Live Graph (IF APPLICABLE/ASKED):** If a visual aids the explanation or the user explicitly asks, you MUST provide executable Python code to generate the graph.\n"
        "   - The code MUST be wrapped in a standard ```python code block.\n"
        "   - You MUST use `ax.plot()` or similar `ax` methods instead of `plt.plot()`.\n"
        "   - Do NOT include `plt.show()`.\n"
        "   - CRITICAL CODE RULE: You MUST explicitly define every single variable you use. Never reference an undefined variable.\n"
        "   - Assume `matplotlib.pyplot as plt`, `numpy as np`, `scipy`, and a predefined `fig, ax = plt.subplots()` are already available in the environment.\n\n"
        f"--- START SOURCE TEXT ---\n{st.session_state.document_text}\n--- END SOURCE TEXT ---"
    )
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing visual and textual context..."):
            answer, model_used = get_ai_response(
                prompt=prompt, 
                system_context=system_instruction, 
                gemini_key=gemini_api_key, 
                groq_key=groq_api_key, 
                or_key=openrouter_api_key,
                image=st.session_state.source_image
            )
            
            clean_answer = re.sub(r'```(?:python)?.*?```', '', answer, flags=re.DOTALL).strip()
            
            st.markdown(clean_answer)
            st.caption(f"⚡ Answered by: {model_used}")
            
            if "```" in answer:
                try:
                    code_block = re.search(r'```(?:python)?(.*?)```', answer, re.DOTALL).group(1).strip()
                    
                    st.divider()
                    st.subheader("📈 Live Visual Representation")
                    
                    fig, ax = plt.subplots()
                    local_environment = {"plt": plt, "fig": fig, "ax": ax, "np": np, "scipy": scipy}
                    exec(code_block, globals(), local_environment)
                    st.pyplot(fig)
                    
                    with st.expander("🔍 View Python Graphing Code (Optional)"):
                        st.code(code_block, language="python")
                    
                except Exception as e:
                    st.error(f"⚠️ Could not render the live graph: {e}")

        st.session_state.messages.append({"role": "assistant", "content": clean_answer, "model": model_used})
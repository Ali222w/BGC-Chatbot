
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text  # Import speech-to-text function
import fitz  # PyMuPDF for capturing screenshots
import pdfplumber  # For searching text in PDF
from datetime import datetime, timedelta  # Needed for chat history timestamps

# UI texts for chat history (you can adjust these as needed)
UI_TEXTS = {
    "English": {
        "new_chat": "New Chat",
        "previous_chats": "Previous Chats",
        "today": "Today",
        "yesterday": "Yesterday"
    },
    "العربية": {
        "new_chat": "دردشة جديدة",
        "previous_chats": "المحادثات السابقة",
        "today": "اليوم",
        "yesterday": "أمس"
    }
}

# -------------------------
# Chat History Functionality
# -------------------------

# Initialize session state for chat history if not already done 
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'chat_memories' not in st.session_state:
    st.session_state.chat_memories = {}

def create_new_chat():
    """إنشاء محادثة جديدة مستقلة تماماً"""
    chat_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    
    # Create new memory instance for this specific chat
    st.session_state.chat_memories[chat_id] = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    
    # Initialize chat but don't show in history until first message
    if chat_id not in st.session_state.chat_history:
        st.session_state.chat_history[chat_id] = {
            'messages': [],
            'timestamp': datetime.now(),
            'first_message': None,  # Start with no title
            'visible': False,       # Hide from chat list initially
            'page_references': "",  # To store page references if any
            'page_screenshots': []  # To store screenshot paths for page references
        }
    st.rerun()
    return chat_id

def update_chat_title(chat_id, message):
    """تحديث عنوان المحادثة"""
    if chat_id in st.session_state.chat_history:
        title = message.strip().replace('\n', ' ')
        title = title[:50] + '...' if len(title) > 50 else title
        st.session_state.chat_history[chat_id]['first_message'] = title
        st.rerun()

def load_chat(chat_id):
    """تحميل محادثة محددة"""
    if chat_id in st.session_state.chat_history:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
        
        # Get or create memory for this specific chat
        if chat_id not in st.session_state.chat_memories:
            st.session_state.chat_memories[chat_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
            # Rebuild memory from this chat's messages
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.session_state.chat_memories[chat_id].chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    st.session_state.chat_memories[chat_id].chat_memory.add_ai_message(msg["content"])
        
        st.rerun()

def format_chat_title(chat):
    """تنسيق عنوان المحادثة"""
    display_text = chat['first_message']
    if display_text:
        display_text = display_text[:50] + '...' if len(display_text) > 50 else display_text
    else:
        display_text = UI_TEXTS[interface_language]['new_chat']
    return display_text

def format_chat_date(timestamp):
    """تنسيق تاريخ المحادثة"""
    today = datetime.now().date()
    chat_date = timestamp.date()
    
    if chat_date == today:
        return UI_TEXTS[interface_language]['today']
    elif chat_date == today - timedelta(days=1):
        return UI_TEXTS[interface_language]['yesterday']
    else:
        return timestamp.strftime('%Y-%m-%d')

# -------------------------
# End of Chat History Functions
# -------------------------

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  
    page_icon="BGC Logo Colored.svg",  
    layout="wide"  
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# PDF Search and Screenshot Class
class PDFSearchAndDisplay:
    def __init__(self):
        pass

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")
        st.title("الإعدادات")
    else:
        apply_css_direction("ltr")
        st.title("Settings")

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        prompt = ChatPromptTemplate.from_messages([
              ("system", """
            You are a helpful assistant for Basrah Gas Company (BGC). Your task is to answer questions based on the provided context about BGC. The context is supplied as a documented resource (e.g., a multi-page manual or database) that is segmented by pages. Follow these rules strictly:

            1. **Language Handling:**
               - If the question is in English, answer in English.
               - If the question is in Arabic, answer in Arabic.
               - If the user explicitly requests a response in a specific language, respond in that language.
               - If the user’s interface language conflicts with the language of the available context, provide the best possible answer in the available language while noting any limitations if needed.

            2. **Understanding and Using Context:**
               - The “provided context” refers to the complete set of documents, manuals, or data provided, segmented into pages.
               - When answering a question, refer only to the relevant section or page of this context.
               - If the question spans multiple sections or pages, determine which page is most directly related to the question. If ambiguity remains, ask the user for clarification.

            3. **Handling Unclear, Ambiguous or Insufficient Information:**
               - If a question is unclear or lacks sufficient context, respond with an appropriate message.
               
            4. **User Interface Language and Content Availability:**
               - Prioritize the user's interface language when formulating answers.
               
            5. **Professional Tone:**
               - Maintain a professional, respectful, and factual tone.
               
            6. **Page-Specific Answers and Source Referencing:**
               - When a question relates directly to content on a specific page, use that page’s content exclusively.
               
            7. **Handling Overlapping or Multiple Relevant Contexts:**
               - If the question might be answered by content on multiple pages, determine the most directly relevant page.
               
            8. **Addressing Updates and Content Discrepancies:**
               - If there are discrepancies due to updates, mention that the answer is based on the latest available context.
               
            9. **Additional Examples and Clarifications:**
               - Always double-check that your answer strictly adheres to the provided context.
               
            10. **Section-Specific Answers and Source Referencing:**
               - Include appropriate references if needed.
               
            By following these guidelines, you will provide accurate, context-based answers while maintaining clarity and professionalism.
"""
),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ])

        if "vectors" not in st.session_state:
            with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                embeddings_path = "embeddings"
                embeddings_path_2 = "embeddingsocr"
                try:
                    vectors_1 = FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
                    vectors_2 = FAISS.load_local(embeddings_path_2, embeddings, allow_dangerous_deserialization=True)
                    vectors_1.merge_from(vectors_2)
                    st.session_state.vectors = vectors_1
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None
        st.markdown("### الإدخال الصوتي" if interface_language == "العربية" else "### Voice Input")
        input_lang_code = "ar" if interface_language == "العربية" else "en"
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language=input_lang_code,
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        if st.button("إعادة تعيين الدردشة" if interface_language == "العربية" else "Reset Chat"):
            st.session_state.messages = []
            # Clear the memory for the current chat
            if st.session_state.current_chat_id in st.session_state.chat_memories:
                st.session_state.chat_memories[st.session_state.current_chat_id].clear()
            st.success("تمت إعادة تعيين الدردشة بنجاح." if interface_language == "العربية" else "Chat has been reset successfully.")
            st.rerun()
    else:
        st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

    # --- Chat History Sidebar ---
    if st.button(UI_TEXTS[interface_language]['new_chat'], use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {UI_TEXTS[interface_language]['previous_chats']}")
    
    chats_by_date = {}
    for chat_id, chat_data in st.session_state.chat_history.items():
        if chat_data['visible'] and chat_data['messages']:
            date = chat_data['timestamp'].date()
            chats_by_date.setdefault(date, []).append((chat_id, chat_data))
    
    for date in sorted(chats_by_date.keys(), reverse=True):
        chats = chats_by_date[date]
        st.markdown(f"#### {format_chat_date(chats[0][1]['timestamp'])}")
        for chat_id, chat_data in sorted(chats, key=lambda x: x[1]['timestamp'], reverse=True):
            if st.sidebar.button(format_chat_title(chat_data), key=f"chat_{chat_id}", use_container_width=True):
                load_chat(chat_id)
    # --- End of Chat History Sidebar ---

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main area for chat interface
col1, col2 = st.columns([1, 4])
with col1:
    st.image("BGC Logo Colored.svg", width=100)
with col2:
    if interface_language == "العربية":
        st.title("بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        هذا بوت الدردشة الخاص بشركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.  
        **كيفية الاستخدام:**  
        - اكتب سؤالك في مربع النص أدناه.  
        - أو استخدم زر المايكروفون للتحدث مباشرة.  
        - سيتم الرد عليك بناءً على المعلومات المتاحة.  
        - تنبيه⚠️: بوت الدردشة هذا لا يزال نموذجًا أوليًا.
        """)
    else:
        st.title("BGC ChatBot")
        st.write("""
        **Welcome!**  
        This is the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.  
        **How to use:**  
        - Type your question in the text box below.  
        - Or use the microphone button to speak directly.  
        - You will receive a response based on the available information.
        """)

# Remove the global memory initialization since we now use per-chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure there is a current chat; if not, create a new one
if st.session_state.current_chat_id is None:
    create_new_chat()

negative_phrases = [
    "I'm sorry",
    "عذرًا",
    "لا أملك معلومات كافية",
    "I don't have enough information",
    "لم أتمكن من فهم سؤالك",
    "I couldn't understand your question",
    "لا يمكنني الإجابة على هذا السؤال",
    "I cannot answer this question",
    "يرجى تقديم المزيد من التفاصيل",
    "Please provide more details",
    "غير واضح",
    "Unclear",
    "غير متأكد",
    "Not sure",
    "لا أعرف",
    "I don't know",
    "غير متاح",
    "Not available",
    "غير موجود",
    "Not found",
    "غير معروف",
    "Unknown",
    "غير محدد",
    "Unspecified",
    "غير مؤكد",
    "Uncertain",
    "غير كافي",
    "Insufficient",
    "غير دقيق",
    "Inaccurate",
    "غير مفهوم",
    "Not clear",
    "غير مكتمل",
    "Incomplete",
    "غير صحيح",
    "Incorrect",
    "غير مناسب",
    "Inappropriate",
    "Please provide me",
    "يرجى تزويدي",
    "Can you provide more",
    "هل يمكنك تقديم المزيد"
]

# Display chat history messages in the main area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Display stored page references (with screenshots) if available ---
current_chat = st.session_state.chat_history.get(st.session_state.current_chat_id, {})
if current_chat.get("page_references"):
    with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References"):
        st.write("هذه الإجابة وفقًا للصفحات: " + current_chat["page_references"]
                 if interface_language == "العربية"
                 else "This answer is according to pages: " + current_chat["page_references"])
        if current_chat.get("page_screenshots"):
            for screenshot in current_chat["page_screenshots"]:
                st.image(screenshot)
# --- End of stored references display ---

# --- Process voice input ---
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    with st.chat_message("user"):
        st.markdown(voice_input)
    
    if len(st.session_state.messages) == 1:
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = voice_input
        st.session_state.chat_history[st.session_state.current_chat_id]['visible'] = True
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = list(st.session_state.messages)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        # Use the current chat's memory for history:
        current_memory = st.session_state.chat_memories[st.session_state.current_chat_id]
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({
            "input": voice_input,
            "context": retriever.get_relevant_documents(voice_input),
            "history": current_memory.chat_memory.messages
        })
        assistant_response = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = list(st.session_state.messages)
        current_memory.chat_memory.add_user_message(voice_input)
        current_memory.chat_memory.add_ai_message(assistant_response)

        if not any(phrase in assistant_response for phrase in negative_phrases):
            page_numbers = set()
            if "context" in response:
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "unknown")
                    if page_number != "unknown" and str(page_number).isdigit():
                        page_numbers.add(int(page_number))
            if page_numbers:
                page_numbers_str = ", ".join(map(str, sorted(page_numbers)))
                screenshots = pdf_searcher.capture_screenshots(pdf_path, [(p, "") for p in page_numbers])
                st.session_state.chat_history[st.session_state.current_chat_id]["page_references"] = page_numbers_str
                st.session_state.chat_history[st.session_state.current_chat_id]["page_screenshots"] = screenshots
                with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References"):
                    st.write("هذه الإجابة وفقًا للصفحات: " + page_numbers_str
                             if interface_language == "العربية"
                             else "This answer is according to pages: " + page_numbers_str)
                    for screenshot in screenshots:
                        st.image(screenshot)
            else:
                st.session_state.chat_history[st.session_state.current_chat_id]["page_references"] = ""
                st.session_state.chat_history[st.session_state.current_chat_id]["page_screenshots"] = []
                with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References"):
                    st.write("لا توجد أرقام صفحات صالحة في السياق." if interface_language == "العربية"
                             else "No valid page numbers available in the context.")
    else:
        assistant_response = (
            "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." 
            if interface_language == "العربية" 
            else "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = list(st.session_state.messages)

# --- Process text input ---
if interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)
    
    if len(st.session_state.messages) == 1:
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = human_input
        st.session_state.chat_history[st.session_state.current_chat_id]['visible'] = True
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = list(st.session_state.messages)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        current_memory = st.session_state.chat_memories[st.session_state.current_chat_id]
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({
            "input": human_input,
            "context": retriever.get_relevant_documents(human_input),
            "history": current_memory.chat_memory.messages
        })
        assistant_response = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = list(st.session_state.messages)
        current_memory.chat_memory.add_user_message(human_input)
        current_memory.chat_memory.add_ai_message(assistant_response)

        if not any(phrase in assistant_response for phrase in negative_phrases):
            page_numbers = set()
            if "context" in response:
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "unknown")
                    if page_number != "unknown" and str(page_number).isdigit():
                        page_numbers.add(int(page_number))
            if page_numbers:
                page_numbers_str = ", ".join(map(str, sorted(page_numbers)))
                screenshots = pdf_searcher.capture_screenshots(pdf_path, [(p, "") for p in page_numbers])
                st.session_state.chat_history[st.session_state.current_chat_id]["page_references"] = page_numbers_str
                st.session_state.chat_history[st.session_state.current_chat_id]["page_screenshots"] = screenshots
                with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References"):
                    st.write("هذه الإجابة وفقًا للصفحات: " + page_numbers_str
                             if interface_language == "العربية"
                             else "This answer is according to pages: " + page_numbers_str)
                    for screenshot in screenshots:
                        st.image(screenshot)
            else:
                st.session_state.chat_history[st.session_state.current_chat_id]["page_references"] = ""
                st.session_state.chat_history[st.session_state.current_chat_id]["page_screenshots"] = []
                with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References"):
                    st.write("لا توجد أرقام صفحات صالحة في السياق." if interface_language == "العربية"
                             else "No valid page numbers available in the context.")
    else:
        assistant_response = (
            "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." 
            if interface_language == "العربية" 
            else "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = list(st.session_state.messages)

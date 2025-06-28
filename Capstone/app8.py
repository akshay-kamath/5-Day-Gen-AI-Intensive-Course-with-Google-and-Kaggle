import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile # For creating temporary files for video uploads
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from PIL import Image
import io
import time # For simulating delays
import httpx # For downloading files in notebook, and potentially for video/image streams
from google.generativeai import types # For File API types
from typing import Optional

# --- Configuration and Setup ---
st.set_page_config(
    page_title="ChetnaShakti AI Coach (Comprehensive)",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🙏 ChetnaShakti: AI-Powered Coach (Comprehensive Demo)")
st.markdown(
    """
    Welcome to ChetnaShakti, your personalized AI wellness coach!
    This demo showcases the comprehensive capabilities of ChetnaShakti, including:
    - **Multi-Agent System:** Interact with CBT, Habit, Spiritual, Journal Analysis, and Information agents.
    - **Contextual Grounding (RAG):** Upload PDFs for spiritual insights & analyze journal entries.
    - **Multimodal Understanding:** Analyze images and **videos**.
    - **Google Search Integration:** Information agent can perform live web searches.
    """
)

# --- API Key Handling ---
# Initialize session state for API key if not already present
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None

# Function to configure the generative AI models
def configure_genai_models(api_key):
    try:
        genai.configure(api_key=api_key)
        # Initialize models only if configuration is successful
        st.session_state.text_model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.video_model = genai.GenerativeModel('gemini-1.5-flash')
        return True
    except Exception as e:
        st.error(f"Error configuring Google API with provided key: {e}. Please check your API key.")
        return False

# Check if a key is already configured and functional
api_key_configured = False
if st.session_state.google_api_key:
    api_key_configured = configure_genai_models(st.session_state.google_api_key)

# If not configured, try loading from secrets or env vars
if not api_key_configured:
    if "GOOGLE_API_KEY" in st.secrets:
        if configure_genai_models(st.secrets["GOOGLE_API_KEY"]):
            st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
            api_key_configured = True
    elif os.environ.get("GOOGLE_API_KEY"):
        if configure_genai_models(os.environ.get("GOOGLE_API_KEY")):
            st.session_state.google_api_key = os.environ.get("GOOGLE_API_KEY")
            api_key_configured = True

# If still not configured, display input field
if not api_key_configured:
    st.warning("Please enter your Google API Key to use the application.")
    api_key_input = st.text_input("Google API Key", type="password", key="api_key_input_widget")
    if api_key_input:
        if configure_genai_models(api_key_input):
            st.session_state.google_api_key = api_key_input
            st.success("API Key set! You can now interact with the coach.")
            st.rerun() # Rerun to activate the rest of the app
        else:
            # If input key fails, ensure it's not stored as valid
            st.session_state.google_api_key = None
    st.stop() # Stop execution if no valid key is present

# Assign models from session state (they are guaranteed to be there if we reach this point)
text_model = st.session_state.text_model
vision_model = st.session_state.vision_model
video_model = st.session_state.video_model

# --- Session State Initialization (rest of session state) ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store_spiritual' not in st.session_state:
    st.session_state.vector_store_spiritual = None
if 'vector_store_journal' not in st.session_state:
    st.session_state.vector_store_journal = None
if 'rag_status_spiritual' not in st.session_state:
    st.session_state.rag_status_spiritual = "Not Loaded"
if 'rag_status_journal' not in st.session_state:
    st.session_state.rag_status_journal = "Not Loaded"
if 'journal_raw_text' not in st.session_state:
    st.session_state.journal_raw_text = ""
if 'spiritual_cache_loaded' not in st.session_state:
    st.session_state.spiritual_cache_loaded = False
# For Gen AI Evaluation metrics
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {
        "cbt": {"helpful": 0, "unhelpful": 0},
        "habit": {"helpful": 0, "unhelpful": 0},
        "spiritual": {"helpful": 0, "unhelpful": 0},
        "journal_analysis": {"helpful": 0, "unhelpful": 0},
        "multimodal_image": {"helpful": 0, "unhelpful": 0},
        "multimodal_video": {"helpful": 0, "unhelpful": 0},
        "information": {"helpful": 0, "unhelpful": 0},
        "total_interactions": 0,
        "total_helpful": 0,
        "total_unhelpful": 0
    }

# --- RAG (Retrieval Augmented Generation) Functions ---
@st.cache_resource
def load_and_process_pdfs(uploaded_files):
    if not uploaded_files:
        return None, "No PDFs uploaded."

    all_docs = []
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary directory if it doesn't exist
            temp_dir = "/tmp/pdf_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            return None, f"Error loading {uploaded_file.name}"

    if not all_docs:
        return None, "No valid documents processed."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    with st.spinner("Creating spiritual knowledge base... This might take a moment."):
        vectorstore = FAISS.from_documents(splits, embeddings)
    st.success("Spiritual knowledge base created successfully!")
    return vectorstore, "Loaded"

@st.cache_resource
def process_journal_text_for_rag(journal_text):
    if not journal_text:
        return None, "No journal text provided."

    from langchain.schema import Document
    doc = Document(page_content=journal_text, metadata={"source": "user_journal"})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents([doc])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    with st.spinner("Analyzing journal entry..."):
        vectorstore = FAISS.from_documents(splits, embeddings)
    st.success("Journal analysis ready!")
    return vectorstore, "Loaded"

# --- Agent Definitions and Prompts with Structured Schemas (Exact from Notebook) ---

# CBT Agent
cbt_prompt_template = """
You are a Cognitive Behavioral Therapy (CBT) AI coach. Your goal is to help the user identify and reframe negative thought patterns.
Analyze the user's statement and provide a structured response in JSON format.
The JSON should have the following keys: "original_thought", "thought_distortion", "challenge", "balanced_thought".

User's statement: "{user_input}"
"""
cbt_schema = {
    "type": "OBJECT",
    "properties": {
        "original_thought": {"type": "STRING"},
        "thought_distortion": {"type": "STRING"},
        "challenge": {"type": "STRING"},
        "balanced_thought": {"type": "STRING"}
    },
    "required": ["original_thought", "thought_distortion", "challenge", "balanced_thought"]
}

# Habit Agent
habit_prompt_template = """
You are a Habit Rewiring AI coach. Your goal is to help the user break down their habit goals into actionable steps based on habit loops (cue, routine, reward).
Analyze the user's statement and provide a structured response in JSON format.
The JSON should have the following keys: "habit_goal", "habit_loop" (an object with "cue", "routine", "reward"), and "tracking_tip".

User's statement: "{user_input}"
"""
habit_schema = {
    "type": "OBJECT",
    "properties": {
        "habit_goal": {"type": "STRING"},
        "habit_loop": {
            "type": "OBJECT",
            "properties": {
                "cue": {"type": "STRING"},
                "routine": {"type": "STRING"},
                "reward": {"type": "STRING"}
            },
            "required": ["cue", "routine", "reward"]
        },
        "tracking_tip": {"type": "STRING"}
    },
    "required": ["habit_goal", "habit_loop", "tracking_tip"]
}

# Spiritual Agent (with RAG context)
spiritual_prompt_template = """
You are a Spiritual Guidance AI coach. Your goal is to provide insights and comfort based on ancient wisdom.
Use the provided context from spiritual texts to answer the user's query. If the context is not relevant, state that you cannot find relevant information in the provided texts and offer general spiritual guidance.
Provide a structured response in JSON format.

Context from spiritual texts:
{context}

User's statement: "{user_input}"
"""
spiritual_schema = {
    "type": "OBJECT",
    "properties": {
        "identified_state": {"type": "STRING"},
        "spiritual_source": {"type": "STRING"},
        "relevant_insight": {"type": "STRING"},
        "interpretation": {"type": "STRING"},
        "daily_mantra": {"type": "STRING"}
    },
    "required": ["identified_state", "spiritual_source", "relevant_insight", "interpretation", "daily_mantra"]
}

# Journal Analysis Agent
journal_analysis_prompt_template = """
You are an AI Journal Analyst. Your task is to analyze the provided journal entry and extract key insights.
Provide a structured response in JSON format with the following keys:
"main_themes": A list of primary topics or recurring ideas.
"emotional_tone": A summary of the overall emotional sentiment (e.g., "reflective," "frustrated," "hopeful").
"potential_cognitive_distortions": A list of any common thinking errors identified (e.g., "all-or-nothing thinking," "catastrophizing," "overgeneralization").
"suggested_reframing": A brief suggestion on how a negative thought or situation could be re-evaluated.
"actionable_insights": A list of 1-2 practical steps or reflections the user could take.

Journal Entry:
{journal_full_text}

User's query/focus: "{user_input}"
"""
journal_analysis_schema = {
    "type": "OBJECT",
    "properties": {
        "main_themes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "emotional_tone": {"type": "STRING"},
        "potential_cognitive_distortions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "suggested_reframing": {"type": "STRING"},
        "actionable_insights": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["main_themes", "emotional_tone", "potential_cognitive_distortions", "suggested_reframing", "actionable_insights"]
}

# Information Agent (uses Google Search)
information_prompt_template = """
You are an Information Agent. Your goal is to provide concise and helpful information based on the provided search results.
If the search results are not sufficient to fully answer the user's query, state that you can only provide information from the available context.

Search Results:
{search_results}

User's query: "{user_input}"
"""

# --- Image Analysis Prompts (Exact from Notebook) ---
scripture_image_analysis_template = """Analyze this image suspected to contain a Sanskrit verse from a scripture like the Bhagavad Gita or Durga Saptashati. The user is currently {user_context}.

Please perform the following steps and structure your response clearly using Markdown:
1.  **Transcription:** Extract the Sanskrit text (in Devanagari script) as accurately as possible. If text is unclear, note that.
2.  **Transliteration:** Provide the standard IAST transliteration of the extracted Sanskrit text.
3.  **Translation:** Provide a clear English translation of the verse.
4.  **Source Identification (Optional):** If possible, identify the likely source (e.g., Bhagavad Gita Chapter X, Verse Y, or Durga Saptashati chapter/context). State if unsure.
5.  **Contextual Relevance:** Briefly explain the meaning or significance of this verse, especially considering the user's context of '{user_context}'.
"""

visual_journal_mood_analysis_template = """Analyze this visual journal image (likely a drawing, mandala, or abstract representation).

Focus on the visual elements:
1.  **Color Palette:** Describe the dominant colors used and their potential emotional connotations (e.g., red=passion/anger, blue=calm/sadness, yellow=optimism/anxiety).
2.  **Shapes/Symbols:** Identify any recurring shapes (e.g., sharp angles, flowing curves, spirals) or recognizable symbols (e.g., lotus, eye, heart). What might they represent in a wellness context?
3.  **Overall Mood/Energy:** Based on the colors, composition, and line quality (if applicable), describe the overall perceived mood or energy (e.g., calm, chaotic, balanced, energetic, heavy, light).
4.  **Potential Theme:** Suggest a possible emotional or spiritual theme the creator might be exploring (e.g., grounding, release of anger, seeking clarity, expressing joy).
5.  **Suggestion:** Offer a brief, relevant suggestion based on the visual analysis, such as a simple mantra or a focus point for reflection (e.g., "Consider the mantra 'Om Shanti' for peace if the image feels chaotic," or "Reflect on the meaning of the spiral for growth").

Structure your response using Markdown headings."""

handwriting_text_analysis_template = """Analyze this image of a handwritten journal entry or affirmation.

Perform the following steps:
1.  **Transcription:** Extract the handwritten text as accurately as possible. If parts are illegible, note that.
2.  **Sentiment/Keywords:** Analyze the transcribed text. Identify the overall sentiment (positive, negative, neutral, mixed) and key emotional keywords or themes (e.g., "anxious," "grateful," "stuck," "letting go," "self-doubt").
3.  **Suggested Action:** Based *only* on the text content, suggest ONE relevant action from the following categories:
    * **CBT:** If negative thoughts are present, suggest a simple CBT technique (e.g., "Identify the core thought," "Look for cognitive distortions like black-and-white thinking," "Challenge the thought").
    * **Mantra:** If seeking peace, strength, or a positive focus, suggest a simple, relevant mantra (e.g., "I am capable," "This feeling will pass," "Om Namah Shivaya").
    * **Habit:** If the text mentions goals or routines, suggest a tiny habit step (e.g., "Focus on the cue for the habit," "Reward yourself after the routine").
    * **Reflection:** Suggest a simple reflection question related to the theme (e.g., "What triggers this feeling?", "What is one small step towards this goal?").

Structure your response using Markdown headings."""


# Video Analysis Prompt (Exact from Notebook)
video_analysis_prompt_template = """You are an AI assistant analyzing a user's video for wellness insights. The user provided this context: "{user_prompt}"

Please analyze the video considering the following aspects and structure your response using Markdown:

1.  **Transcription:** Provide a summary or key phrases transcribed from the spoken content. If audio is unclear or absent, state that.
2.  **Emotional Analysis (Multimodal):**
    * **Vocal Tone:** Describe the perceived tone of voice (e.g., calm, anxious, energetic, fatigued, flat, distressed).
    * **Facial Expression (if visible):** Describe any discernible facial expressions related to emotion (e.g., smiling, frowning, showing tension, neutral). If the face is not clearly visible, state that.
    * **Overall Sentiment:** Based on transcription, tone, and expression (where available), assess the likely overall emotional sentiment (e.g., positive, negative, neutral, mixed, leaning towards frustration, expressing hope, showing vulnerability).
3.  **Key Themes/Keywords:** Identify the main topics or recurring keywords mentioned in the transcription (e.g., "overwhelmed," "grateful," "stuck," "relationship," "sleep," "purpose").
4.  **Wellness Suggestions (Grounded in Analysis):** Based on the combined analysis above (sentiment, themes), provide 1-2 relevant suggestions drawing from these areas:
    * **CBT:** If negative thought patterns seem present (based on transcription/sentiment), suggest a simple reframing question or a focus point (e.g., "Consider asking: 'Is this thought 100% true?'", "Notice if 'all-or-nothing' thinking is present.").
    * **Habit:** If the user mentions goals, routines, or struggles with action, suggest a tiny habit idea (e.g., "Try linking [desired action] to an existing cue like brushing teeth," "Focus on just 2 minutes of [activity]").
    * **Spiritual:** If themes of meaning, peace, strength, or specific spiritual language arise, suggest a relevant concept, mantra, or reflection point (e.g., "Reflect on the concept of acceptance," "Consider the mantra 'Om Shanti' for peace," "Explore the Durga Saptashati verse 'Ya Devi Sarva Bhuteshu...' for inner strength if seeking resilience").
    * **Mindfulness/Somatic:** If stress or overwhelm is evident, suggest a brief grounding or breathing exercise (e.g., "Try a 1-minute body scan," "Practice 4-7-8 breathing").

**Important:** Clearly state the basis for each suggestion (e.g., "Based on the theme of feeling overwhelmed, consider..."). Acknowledge limitations if audio/video quality hinders analysis.
"""
video_analysis_disclaimer = """
*Disclaimer: AI analysis of emotion and sentiment from video is experimental and may not be fully accurate. This is not a substitute for professional medical or psychological diagnosis or or advice. Please consult with a qualified professional for personal health concerns.*
"""


# --- Gemini API Call Functions ---
def get_structured_ai_response(prompt_text, response_schema, model_to_use=text_model):
    try:
        response = model_to_use.generate_content(
            [{"role": "user", "parts": [{"text": prompt_text}]}],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )
        json_string = response.candidates[0].content.parts[0].text
        return json.loads(json_string)
    except Exception as e:
        st.error(f"AI Response Error: {e}. The model might not have generated valid JSON for the schema. Please try again or refine your query.")
        return {"error": "Could not generate structured response. Please try again or refine your query."}

def get_plain_ai_response(prompt_text, model_to_use=text_model):
    try:
        response = model_to_use.generate_content([{"role": "user", "parts": [{"text": prompt_text}]}])
        return response.text
    except Exception as e:
        st.error(f"AI Response Error: {e}")
        return "Could not generate response. Please try again."

def analyze_image(image_data, analysis_type, user_context):
    try:
        image_part = {
            "mime_type": "image/jpeg", # Assuming JPEG, adjust if needed
            "data": image_data
        }
        
        prompt = ""
        if analysis_type == "Scripture Analysis":
            prompt = scripture_image_analysis_template.format(user_context=user_context)
        elif analysis_type == "Visual Journal Analysis (Mood/Symbols)":
            prompt = visual_journal_mood_analysis_template
        elif analysis_type == "Handwriting Analysis (Text/Sentiment)":
            prompt = handwriting_text_analysis_template

        response = vision_model.generate_content([prompt, image_part])
        return response.text
    except Exception as e:
        st.error(f"Image Analysis Error: {e}")
        return "Could not process image. Please try again."

# --- Video Processing Functions (from notebook) ---
def upload_and_process_video_gemini(file_bytes: bytes, file_name: str) -> Optional[types.File]:
    """Uploads a video file (from bytes) to Gemini File API and waits for it to be processed."""
    
    # Create a temporary file to save the uploaded bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
        tmp_file.write(file_bytes)
        temp_file_path = tmp_file.name

    st.info(f"Uploading video '{file_name}' to Gemini File API...")
    try:
        video_file = genai.upload_file(path=temp_file_path)
        st.success(f"Upload complete. File ID: {video_file.name}")
    except Exception as e:
        st.error(f"Error during video upload: {e}")
        os.unlink(temp_file_path) # Clean up temp file
        return None

    st.info("Waiting for video processing... (This may take several minutes for longer videos)")
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    # Simulate progress for user feedback
    # This loop is for visual feedback only, actual processing time is unknown
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Processing... {i+1}% complete. Elapsed: {int(time.time() - start_time)}s")
        time.sleep(0.5) # Small delay for visual effect

    # Actual polling for processing status
    while video_file.state.name == "PROCESSING":
        time.sleep(5) # Check every 5 seconds
        try:
            video_file = genai.get_file(name=video_file.name)
        except Exception as e:
            st.warning(f"Error getting file status: {e}. Retrying...")
            time.sleep(10) # Wait longer before retrying
    
    progress_bar.empty() # Clear progress bar
    status_text.empty() # Clear status text

    if video_file.state.name == "FAILED":
        st.error(f"Video processing failed: {video_file.name}")
        try:
            genai.delete_file(name=video_file.name)
            st.info(f"Cleaned up failed file resource: {video_file.name}")
        except Exception as e_del:
            st.warning(f"Warning: Could not delete failed file resource {video_file.name}: {e_del}")
        os.unlink(temp_file_path) # Clean up temp file
        return None
    elif video_file.state.name == "ACTIVE":
        st.success(f"Video processing complete. File URI: {video_file.uri}")
        os.unlink(temp_file_path) # Clean up temp file
        return video_file
    else:
        st.error(f"Unexpected video file state: {video_file.state.name}")
        os.unlink(temp_file_path) # Clean up temp file
        return None

def analyze_wellness_video_gemini(video_file: types.File, user_prompt: str) -> str:
    """Analyzes a wellness-related video using the Gemini model."""
    if not video_file or video_file.state.name != "ACTIVE":
        return "Error: Invalid or unprocessed video file provided." + video_analysis_disclaimer

    st.info(f"Analyzing wellness video: {video_file.name}")

    try:
        contents = [video_analysis_prompt_template.format(user_prompt=user_prompt), video_file]
        response = video_model.generate_content(contents=contents)
        return response.text + video_analysis_disclaimer
    except Exception as e:
        st.error(f"Error generating content from video: {e}")
        if "quota" in str(e).lower():
             return "Sorry, I couldn't analyze the video due to current usage limits. Please try again later." + video_analysis_disclaimer
        return f"An error occurred during video analysis: {e}" + video_analysis_disclaimer

# --- Intent Routing (Simulating LangGraph's Conditional Edges/Routing) ---
def classify_intent(query, chat_history, has_uploaded_video=False):
    query_lower = query.lower()
    
    # 1. Explicit Agent Commands (Highest Priority)
    if "cbt agent" in query_lower or "use cbt" in query_lower or "reframe my thought" in query_lower:
        return "cbt"
    if "habit agent" in query_lower or "use habit coach" in query_lower or "help with habit" in query_lower:
        return "habit"
    if "spiritual agent" in query_lower or "use spiritual guidance" in query_lower or "ask gita" in query_lower or "ask scripture" in query_lower:
        return "spiritual"
    if "analyze journal" in query_lower or "my journal" in query_lower or "journal entry analysis" in query_lower:
        return "journal_analysis"
    if "describe image" in query_lower or "analyze image" in query_lower or "what is in this picture" in query_lower:
        return "multimodal_image"
    if ("video" in query_lower and ("understand" in query_lower or "analyze" in query_lower or "process" in query_lower)) or has_uploaded_video:
        return "multimodal_video" # Route to video agent if video is uploaded or explicitly asked
    if "search for" in query_lower or "google search" in query_lower or "find information about" in query_lower or "tell me about" in query_lower:
        return "information"

    # 2. Contextual Routing (Based on previous turn, simple memory)
    if chat_history and chat_history[-1]["role"] == "model":
        last_response_content = chat_history[-1]["content"]
        if isinstance(last_response_content, dict):
            if "original_thought" in last_response_content: return "cbt"
            if "habit_goal" in last_response_content: return "habit"
            if "identified_state" in last_response_content: return "spiritual"
            if "main_themes" in last_response_content: return "journal_analysis"
        elif "Image Analysis:" in last_response_content: return "multimodal_image"
        elif "Sources:" in last_response_content or "Sorry, an error occurred while performing the search" in last_response_content: return "information"
        elif "Video Analysis:" in last_response_content: return "multimodal_video"


    # 3. Keyword-based Routing (General Classification)
    if any(keyword in query_lower for keyword in ["feeling", "anxiety", "stress", "overwhelmed", "depressed", "mood", "emotion"]):
        return "cbt"
    if any(keyword in query_lower for keyword in ["goal", "routine", "daily", "improve", "change", "start", "stop", "habit"]):
        return "habit"
    if any(keyword in query_lower for keyword in ["meaning", "purpose", "wisdom", "enlightenment", "divine", "spiritual", "scripture"]):
        return "spiritual"
    
    # Default to Information Agent for unclassified queries not caught by specific agents
    return "information"

# --- Structured Output Rendering Functions ---
def render_cbt_response(response_dict):
    markdown_output = f"""
    ### 🧠 CBT Analysis
    **Original Thought:** {response_dict.get('original_thought', 'N/A')}
    **Thought Distortion:** {response_dict.get('thought_distortion', 'N/A')}
    **Challenge:** {response_dict.get('challenge', 'N/A')}
    **Balanced Thought:** {response_dict.get('balanced_thought', 'N/A')}
    """
    return markdown_output

def render_habit_response(response_dict):
    habit_loop = response_dict.get('habit_loop', {})
    markdown_output = f"""
    ### 🎯 Habit Rewiring Plan
    **Habit Goal:** {response_dict.get('habit_goal', 'N/A')}
    **Habit Loop:**
    * **Cue:** {habit_loop.get('cue', 'N/A')}
    * **Routine:** {habit_loop.get('routine', 'N/A')}
    * **Reward:** {habit_loop.get('reward', 'N/A')}
    **Tracking Tip:** {response_dict.get('tracking_tip', 'N/A')}
    """
    return markdown_output

def render_spiritual_response(response_dict):
    markdown_output = f"""
    ### ✨ Spiritual Insight
    **Identified State:** {response_dict.get('identified_state', 'N/A')}
    **Spiritual Source:** {response_dict.get('spiritual_source', 'N/A')}
    **Relevant Insight:** {response_dict.get('relevant_insight', 'N/A')}
    **Interpretation:** {response_dict.get('interpretation', 'N/A')}
    **Daily Mantra:** {response_dict.get('daily_mantra', 'N/A')}
    """
    return markdown_output

def render_journal_analysis_response(response_dict):
    main_themes = ", ".join(response_dict.get('main_themes', ['N/A']))
    potential_distortions = ", ".join(response_dict.get('potential_cognitive_distortions', ['None identified']))
    actionable_insights = "\n".join([f"* {item}" for item in response_dict.get('actionable_insights', ['N/A'])])

    markdown_output = f"""
    ### 📝 Journal Analysis
    **Main Themes:** {main_themes}
    **Emotional Tone:** {response_dict.get('emotional_tone', 'N/A')}
    **Potential Cognitive Distortions:** {potential_distortions}
    **Suggested Reframing:** {response_dict.get('suggested_reframing', 'N/A')}
    **Actionable Insights:**
    {actionable_insights}
    """
    return markdown_output

# --- Feedback Handler for Gen AI Evaluation ---
def handle_feedback(agent_type, is_helpful, message_index):
    if is_helpful:
        st.session_state.evaluation_metrics[agent_type]["helpful"] += 1
        st.session_state.evaluation_metrics["total_helpful"] += 1
    else:
        st.session_state.evaluation_metrics[agent_type]["unhelpful"] += 1
        st.session_state.evaluation_metrics["total_unhelpful"] += 1
    st.session_state.evaluation_metrics["total_interactions"] += 1
    
    # Mark this message as having received feedback to prevent multiple clicks
    if 'feedback_given' not in st.session_state.chat_history[message_index]:
        st.session_state.chat_history[message_index]['feedback_given'] = True
    
    st.rerun() # Rerun to update metrics display

# --- Sidebar for Knowledge & Multimodal Inputs (Cleaned UI) ---
with st.sidebar:
    st.header("📚 Knowledge & Inputs")
    st.markdown("Upload relevant files or paste text to enhance ChetnaShakti's understanding.")

    st.subheader("Spiritual Texts (PDFs)")
    st.markdown("Upload sacred texts for the Spiritual Agent to draw insights from (RAG).")
    uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader")
    if uploaded_pdfs:
        if st.button("Process Spiritual PDFs"):
            st.session_state.vector_store_spiritual, st.session_state.rag_status_spiritual = load_and_process_pdfs(uploaded_pdfs)
    st.info(f"Spiritual RAG Status: **{st.session_state.rag_status_spiritual}**")

    st.subheader("Journal Entry Analysis")
    st.markdown("Paste your journal entry for AI-powered thematic and emotional analysis.")
    journal_text_input = st.text_area("Your Journal Entry", height=200, key="journal_text_input", value=st.session_state.journal_raw_text, help="Paste your thoughts, feelings, or daily reflections here. The AI can then analyze it for themes, emotions, and cognitive distortions.")
    
    # Only process if text has changed or if it's the first time and text is present
    if journal_text_input != st.session_state.journal_raw_text:
        st.session_state.journal_raw_text = journal_text_input
        if journal_text_input:
            st.session_state.vector_store_journal, st.session_state.rag_status_journal = process_journal_text_for_rag(journal_text_input)
            st.session_state.chat_history.append({"role": "system", "content": "Journal entry processed. You can now ask the AI to analyze it (e.g., 'Analyze my journal for themes')."})
        else:
            st.session_state.vector_store_journal = None
            st.session_state.rag_status_journal = "Not Loaded"
            st.session_state.chat_history.append({"role": "system", "content": "Journal entry cleared."})
    st.info(f"Journal RAG Status: **{st.session_state.rag_status_journal}**")


    st.subheader("Image Understanding")
    st.markdown("Upload an image for the AI to describe or analyze.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    image_analysis_type = st.radio(
        "Select Image Analysis Type:",
        ("Scripture Analysis", "Visual Journal Analysis (Mood/Symbols)", "Handwriting Analysis (Text/Sentiment)"),
        key="image_analysis_type_radio"
    )
    image_context_prompt = st.text_input("Optional: Provide context for image (e.g., 'I'm feeling overwhelmed by duties' for scripture, or 'This is my mood board')", value="", key="image_context_prompt_input")


    if uploaded_image and st.button("Analyze Image"):
        image_bytes = uploaded_image.getvalue()
        st.session_state.chat_history.append({"role": "user", "content": f"User uploaded an image for '{image_analysis_type}'. Context: '{image_context_prompt}'"})
        st.session_state.chat_history.append({"role": "system", "content": "Analyzing image..."})
        with st.spinner("Analyzing image..."):
            description = analyze_image(image_bytes, image_analysis_type, image_context_prompt)
            st.session_state.chat_history.append({"role": "model", "content": f"**Image Analysis ({image_analysis_type}):**\n\n{description}"})
            st.rerun() # Rerun to update chat history

    st.subheader("Video Understanding")
    st.markdown("Upload a video for AI analysis (e.g., spoken journal, check-in).")
    st.warning("Video analysis can be slow and consume significant API quota. Use small videos for testing.")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "webm"], key="video_uploader")
    video_analysis_user_prompt = st.text_input("Context for video (e.g., 'This is my daily check-in')", value="Analyze my spoken journal entry for emotional state and suggest relevant wellness practices.", key="video_prompt_input")

    if uploaded_video and st.button("Analyze Video"):
        video_bytes = uploaded_video.getvalue()
        video_filename = uploaded_video.name
        st.session_state.chat_history.append({"role": "user", "content": f"User uploaded video '{video_filename}'. Context: '{video_analysis_user_prompt}'"})
        st.session_state.chat_history.append({"role": "system", "content": "Processing video for analysis..."})
        
        with st.spinner("Uploading and processing video..."):
            processed_video_file = upload_and_process_video_gemini(video_bytes, video_filename)

        if processed_video_file:
            with st.spinner("Analyzing video content..."):
                video_analysis_result = analyze_wellness_video_gemini(processed_video_file, video_analysis_user_prompt)
                st.session_state.chat_history.append({"role": "model", "content": f"Video Analysis: {video_analysis_result}"})
            
            # Attempt to delete the file from Gemini File API after analysis
            try:
                genai.delete_file(name=processed_video_file.name)
                st.info(f"Successfully deleted processed video file from Gemini API: {processed_video_file.name}")
            except Exception as e_del:
                st.warning(f"Warning: Could not delete processed video file {processed_video_file.name}: {e_del}")
        else:
            st.session_state.chat_history.append({"role": "model", "content": "Video analysis could not be completed due to processing issues."})
        
        st.rerun() # Rerun to update chat history


# --- Main Chat Interface ---
st.header("Chat with ChetnaShakti Agents")
st.markdown("Type your query below. The AI will intelligently route it to the most relevant agent (CBT, Habit, Spiritual, Journal Analysis, Information, or Multimodal).")

# Display chat history
chat_display_area = st.container(height=500, border=True)
with chat_display_area:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "model":
            # Determine which agent provided this response for feedback tracking
            agent_type_for_feedback = "general" # Default
            if isinstance(message["content"], dict):
                if "original_thought" in message["content"]: agent_type_for_feedback = "cbt"
                elif "habit_goal" in message["content"]: agent_type_for_feedback = "habit"
                elif "identified_state" in message["content"]: agent_type_for_feedback = "spiritual"
                elif "main_themes" in message["content"]: agent_type_for_feedback = "journal_analysis"
                
                # Render structured JSON beautifully
                with st.chat_message("assistant"):
                    if agent_type_for_feedback == "cbt":
                        st.markdown(render_cbt_response(message["content"]))
                    elif agent_type_for_feedback == "habit":
                        st.markdown(render_habit_response(message["content"]))
                    elif agent_type_for_feedback == "spiritual":
                        st.markdown(render_spiritual_response(message["content"]))
                    elif agent_type_for_feedback == "journal_analysis":
                        st.markdown(render_journal_analysis_response(message["content"]))
                    else: # Fallback just in case a dict isn't caught by a specific renderer
                        st.json(message["content"]) 
            else:
                # Handle plain text responses (Image, Video, Information, General)
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    if "Image Analysis:" in message["content"]: agent_type_for_feedback = "multimodal_image"
                    elif "Video Analysis:" in message["content"]: agent_type_for_feedback = "multimodal_video"
                    elif "Sources:" in message["content"] or "Sorry, an error occurred while performing the search" in message["content"]: agent_type_for_feedback = "information"
            
            # Add feedback buttons only for AI responses and if feedback hasn't been given
            if not message.get('feedback_given', False):
                col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                with col1:
                    if st.button("👍 Yes", key=f"helpful_{i}"):
                        handle_feedback(agent_type_for_feedback, True, i)
                with col2:
                    if st.button("👎 No", key=f"unhelpful_{i}"):
                        handle_feedback(agent_type_for_feedback, False, i)
                with col3:
                    st.caption("Was this helpful?")

        elif message["role"] == "system":
            st.info(message["content"]) # Use info box for system messages

# User input
user_input = st.chat_input("Ask ChetnaShakti...", key="chat_input_main")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with chat_display_area:
        st.chat_message("user").write(user_input)

    # Determine intent, passing whether a video was uploaded in this turn
    intent = classify_intent(user_input, st.session_state.chat_history, has_uploaded_video=(uploaded_video is not None))
    
    with st.spinner(f"ChetnaShakti ({intent.replace('_', ' ').title()} Agent) is thinking..."):
        ai_response_content = None

        if intent == "cbt":
            prompt = cbt_prompt_template.format(user_input=user_input)
            ai_response_content = get_structured_ai_response(prompt, cbt_schema)

        elif intent == "habit":
            prompt = habit_prompt_template.format(user_input=user_input)
            ai_response_content = get_structured_ai_response(prompt, habit_schema)

        elif intent == "spiritual":
            context = ""
            if st.session_state.vector_store_spiritual and st.session_state.rag_status_spiritual == "Loaded":
                docs = st.session_state.vector_store_spiritual.similarity_search(user_input, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                if not context: # If no relevant context found, inform the user
                    context = "No highly relevant spiritual text found for this query in the uploaded documents."
            else:
                context = "No spiritual texts have been loaded. Spiritual guidance will be general."
            
            prompt = spiritual_prompt_template.format(context=context, user_input=user_input)
            ai_response_content = get_structured_ai_response(prompt, spiritual_schema)

        elif intent == "journal_analysis":
            if st.session_state.journal_raw_text and st.session_state.rag_status_journal == "Loaded":
                journal_full_text = st.session_state.journal_raw_text
                prompt = journal_analysis_prompt_template.format(journal_full_text=journal_full_text, user_input=user_input)
                ai_response_content = get_structured_ai_response(prompt, journal_analysis_schema)
            else:
                ai_response_content = "Please paste a journal entry in the sidebar first to enable journal analysis."

        elif intent == "multimodal_image":
            ai_response_content = "To analyze an image, please use the 'Image Understanding' section in the sidebar. After uploading and selecting the analysis type, click 'Analyze Image'."
        
        elif intent == "multimodal_video":
            # This branch is primarily for when the user asks about video understanding in chat
            # The actual video processing is triggered by the button in the sidebar.
            ai_response_content = """
            You've asked about video understanding! ChetnaShakti can analyze video inputs for emotional state, vocal tone, and key themes.
            To use this feature, please:
            1.  **Upload a video file** using the 'Choose a video...' button in the sidebar under 'Video Understanding'.
            2.  **Provide context** for the video in the text input below the uploader.
            3.  Click the **'Analyze Video' button**.
            
            Be aware that video analysis can take some time and consumes API quota.
            """

        elif intent == "information":
            # Use actual Google Search via Gemini API
            search_query = user_input
            final_content = ""
            
            try:
                # This is the direct call to Gemini with the Google Search tool, as per your notebook
                # This requires the tool to be available in the execution environment (e.g., Google AI Studio, Vertex AI)
                config_with_search = genai.types.GenerationConfig(
                    tools=[genai.tool_named('google_search')] 
                )
                
                response = text_model.generate_content(
                    contents=search_query, # User query directly as content for tool use
                    generation_config=config_with_search
                )
                
                markdown_buffer = io.StringIO()
                rc = response.candidates[0]

                # Check for grounding metadata first (preferred for search results)
                if hasattr(rc, 'grounding_metadata') and rc.grounding_metadata and rc.grounding_metadata.grounding_chunks:
                    chunks = rc.grounding_metadata.grounding_chunks
                    supports = rc.grounding_metadata.grounding_supports

                    if rc.content and rc.content.parts:
                        response_text_content = rc.content.parts[0].text
                        processed_text = ""
                        last_index = 0
                        if supports:
                            for support in supports:
                                start = support.segment.start_index
                                end = support.segment.end_index
                                processed_text += response_text_content[last_index:start]
                                processed_text += response_text_content[start:end]
                                for i in support.grounding_chunk_indices:
                                    if 0 <= i < len(chunks):
                                        processed_text += f"<sup>[{i+1}]</sup>"
                                last_index = end
                            processed_text += response_text_content[last_index:]
                        else:
                            processed_text = response_text_content
                        markdown_buffer.write(processed_text)
                    
                    markdown_buffer.write("\n\n**Sources:**\n\n")
                    for i, chunk in enumerate(chunks, start=1):
                        if hasattr(chunk, 'web') and chunk.web and chunk.web.uri:
                            title = chunk.web.title if chunk.web.title else f"Source {i}"
                            markdown_buffer.write(f"{i}. [{title}]({chunk.web.uri})\n")
                        else:
                            markdown_buffer.write(f"{i}. (Source information unavailable)\n")
                    
                    final_content = markdown_buffer.getvalue()
                elif rc.content and rc.content.parts:
                    # If no grounding metadata but a direct text response (e.g., model answered without search)
                    final_content = rc.content.parts[0].text
                else:
                    final_content = "No specific information found."

            except Exception as e:
                # Catch specific errors related to tool unavailability in local environment
                # The error might manifest as a tool execution error or a direct API error
                if "tool" in str(e).lower() or "function_call" in str(e).lower() or "google_search" in str(e).lower():
                    final_content = (
                        "**Google Search (Simulated for Local Demo):**\n\n"
                        f"The AI attempted to search for '{search_query}', but the `google_search` tool is typically available "
                        "only in specific Google-managed environments (like Google AI Studio or Vertex AI). "
                        "For local execution with actual web search, you would need to integrate a separate search API (e.g., Google Custom Search API, SerpApi) "
                        "and update the `information` agent to use that external API. "
                        "Here's a *simulated* response for demonstration purposes:\n\n"
                        f"**Simulated Search Result for '{search_query}':** "
                        "Mindfulness practices, such as meditation and deep breathing, are widely recognized for their effectiveness in reducing stress and anxiety. "
                        "Studies suggest that regular practice can lead to improved emotional regulation and cognitive flexibility. "
                        "For more detailed information, you would typically see citations to various web sources here if the search were live."
                    )
                elif "quota" in str(e).lower():
                    final_content = "Sorry, I'm currently unable to perform searches due to usage limits. Please try again later."
                else:
                    final_content = f"Sorry, an unexpected error occurred while performing the search: {e}. Please try again."
            
            ai_response_content = final_content + "\n\n*Disclaimer: Information provided is from web search (or simulated) and is not medical or psychological advice.*"

        else: # Fallback / General Agent (should be rare with good intent classification)
            ai_response_content = get_plain_ai_response(f"I'm ChetnaShakti. How can I help you with '{user_input}'? You can ask about CBT, habits, spiritual guidance, analyze a journal, analyze an image or video (via sidebar), or ask general questions.")

        if ai_response_content:
            st.session_state.chat_history.append({"role": "model", "content": ai_response_content, "agent_type": intent}) # Store agent type for feedback
            with chat_display_area:
                # Re-render the new message (and feedback buttons if applicable)
                if isinstance(ai_response_content, dict):
                    if intent == "cbt":
                        st.markdown(render_cbt_response(ai_response_content))
                    elif intent == "habit":
                        st.markdown(render_habit_response(ai_response_content))
                    elif intent == "spiritual":
                        st.markdown(render_spiritual_response(ai_response_content))
                    elif intent == "journal_analysis":
                        st.markdown(render_journal_analysis_response(ai_response_content))
                    else:
                        st.json(ai_response_content)
                else:
                    st.markdown(ai_response_content)
                
                # Add feedback buttons for the newly added message
                if not st.session_state.chat_history[-1].get('feedback_given', False):
                    col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                    with col1:
                        if st.button("👍 Yes", key=f"helpful_{len(st.session_state.chat_history)-1}"):
                            handle_feedback(intent, True, len(st.session_state.chat_history)-1)
                    with col2:
                        if st.button("👎 No", key=f"unhelpful_{len(st.session_state.chat_history)-1}"):
                            handle_feedback(intent, False, len(st.session_state.chat_history)-1)
                    with col3:
                        st.caption("Was this helpful?")

        # After processing, ensure the chat display scrolls to the bottom
        st.experimental_fragment_rerun() # Use fragment rerun if available for smoother scroll

# --- Gen AI Evaluation Section (from Notebook - Exact Content) ---
st.markdown("---")
with st.expander("📈 **Gen AI Evaluation: Ensuring ChetnaShakti's Effectiveness**"):
    st.markdown(
        """
        In the context of ChetnaShakti, continuous evaluation is crucial to ensure the AI coach remains effective, empathetic, and aligned with user needs. This involves tracking various metrics and gathering feedback.

        ### **1. Emotional Awareness & Cognitive Shifts:**

        * **Metrics:**
            * Sentiment analysis of user inputs over time (e.g., pre-session vs. post-session sentiment scores).
            * Frequency of identified cognitive distortions (e.g., "all-or-nothing thinking," "catastrophizing").
            * Successful reframing rates (how often the user acknowledges a reframed thought as helpful).
            * User self-reports on emotional state changes (e.g., "I feel less anxious now").

        * **Methodology:**
            * AI models can analyze user's emotional tone and identify shifts after interacting with the CBT agent.
            * Post-interaction surveys or quick check-ins asking users to rate their emotional state.
            * Analysis of journal entries for recurring negative patterns and their reduction over time.

        ### **2. Habit Formation & Goal Completion:**

        * **Metrics:**
            * User-reported adherence to new habits (e.g., "I meditated for 10 minutes today").
            * Completion rates of micro-goals set with the Habit Agent.
            * Progress towards larger habit transformation goals (e.g., "I've consistently woken up early for a month").

        * **Methodology:**
            * Simple in-app tracking mechanisms where users can log their habit progress.
            * Periodic check-ins by the AI to inquire about habit adherence.
            * Analysis of user inputs for mentions of successful habit implementation or struggles.

        ### **3. Spiritual Evolution & Insight:**

        * **Metrics:**
            * User satisfaction with spiritual guidance (e.g., ratings of insightfulness, relevance).
            * Perceived depth of insights (qualitative assessment of user feedback).
            * Frequency of engaging with suggested spiritual practices (if tracked).
            * Analysis of journal entries for themes of spiritual growth, peace, or understanding.

        ### **4. Overall User Engagement & Satisfaction:**

        * **Metrics:**
            * Session duration and frequency of use.
            * User retention over time.
            * Direct user ratings and feedback on the overall coaching experience.
            * Completion of multi-turn conversations or specific coaching pathways.

        * **Methodology:**
            * In-app analytics for usage patterns.
            * NPS (Net Promoter Score) or CSAT (Customer Satisfaction) surveys.
            * A/B testing of different interaction flows or agent behaviors.

        ### **5. AI Performance Metrics:**

        * **Metrics:**
            * **Latency:** Response time of AI agents.
            * **Token Usage:** Efficiency of prompt and response generation.
            * **Model Accuracy:** In intent classification (how often the correct agent is chosen).
            * **Relevance of RAG Retrieval:** How well retrieved documents match user queries.
            * **Hallucination Rate:** How often the AI generates factually incorrect or unsupported information.

        * **Methodology:**
            * Monitoring API calls and server logs.
            * Human evaluation of AI responses against a rubric (e.g., factual correctness, helpfulness, empathy).
            * Automated tests for intent classification accuracy.
            * Fine-tuning models based on identified performance gaps.

        This comprehensive evaluation framework ensures ChetnaShakti continuously improves and provides the most impactful support to its users, aligning with the vision of a "conscious agent with intent."
        """
    )

    st.subheader("Current Session Feedback Metrics:")
    total_interactions = st.session_state.evaluation_metrics["total_interactions"]
    total_helpful = st.session_state.evaluation_metrics["total_helpful"]
    total_unhelpful = st.session_state.evaluation_metrics["total_unhelpful"]

    st.markdown(f"**Total AI Interactions:** `{total_interactions}`")
    st.markdown(f"**Total Helpful Responses:** `{total_helpful}`")
    st.markdown(f"**Total Unhelpful Responses:** `{total_unhelpful}`")
    if total_interactions > 0:
        st.markdown(f"**Helpfulness Rate:** `{total_helpful / total_interactions:.2%}`")
    else:
        st.markdown(f"**Helpfulness Rate:** `N/A`")

    st.markdown("---")
    st.markdown("**Feedback per Agent Type:**")
    for agent, metrics in st.session_state.evaluation_metrics.items():
        if isinstance(metrics, dict) and "helpful" in metrics: # Filter out total_interactions etc.
            agent_total = metrics["helpful"] + metrics["unhelpful"]
            if agent_total > 0:
                st.markdown(f"- **{agent.replace('_', ' ').title()} Agent:** Helpful: `{metrics['helpful']}` | Unhelpful: `{metrics['unhelpful']}` | Rate: `{metrics['helpful'] / agent_total:.2%}`")
            else:
                st.markdown(f"- **{agent.replace('_', ' ').title()} Agent:** No feedback yet.")




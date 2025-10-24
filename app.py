import streamlit as st
import os
import json
from datetime import datetime
from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from pydantic import SecretStr
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
import PyPDF2
import io

# Configure page
st.set_page_config(
    page_title="Context-Aware Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

if 'show_clear_confirm' not in st.session_state:
    st.session_state.show_clear_confirm = False

if 'selected_prompt_style' not in st.session_state:
    st.session_state.selected_prompt_style = "Assistant"

if 'conversation_stats' not in st.session_state:
    st.session_state.conversation_stats = {
        'start_time': None,
        'total_user_messages': 0,
        'total_assistant_messages': 0,
        'total_characters': 0,
        'estimated_tokens': 0,
        'response_times': []
    }

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

if 'document_context' not in st.session_state:
    st.session_state.document_context = ""

# Define system prompt styles
SYSTEM_PROMPTS = {
    "Assistant": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user questions. Be friendly and professional in your tone.",

    "Teacher": "You are a knowledgeable teacher and mentor. Explain concepts clearly, provide educational examples, and encourage learning. Break down complex topics into understandable parts and ask follow-up questions to ensure comprehension.",

    "Creative Writer": "You are a creative writing assistant. Help with storytelling, character development, plot ideas, and creative writing techniques. Be imaginative, inspiring, and encourage creative expression.",

    "Technical Expert": "You are a technical expert and programmer. Provide precise technical information, code examples, and detailed explanations for technical problems. Focus on accuracy, best practices, and practical solutions.",

    "Life Coach": "You are a supportive life coach. Help users set goals, overcome challenges, and develop positive habits. Be encouraging, empathetic, and provide actionable advice for personal growth.",

    "Scientist": "You are a scientist and researcher. Provide evidence-based information, explain scientific concepts clearly, and encourage critical thinking. Reference reliable sources and explain the scientific method when appropriate."
}

def estimate_tokens(text):
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def update_conversation_stats(user_message, assistant_message, response_time):
    """Update conversation statistics"""
    if st.session_state.conversation_stats['start_time'] is None:
        st.session_state.conversation_stats['start_time'] = datetime.now()

    stats = st.session_state.conversation_stats

    # Update message counts
    stats['total_user_messages'] += 1
    stats['total_assistant_messages'] += 1

    # Update character counts
    user_chars = len(user_message)
    assistant_chars = len(assistant_message)
    stats['total_characters'] += user_chars + assistant_chars

    # Update token estimates
    user_tokens = estimate_tokens(user_message)
    assistant_tokens = estimate_tokens(assistant_message)
    stats['estimated_tokens'] += user_tokens + assistant_tokens

    # Track response time
    stats['response_times'].append(response_time)

    # Keep only last 10 response times for average calculation
    if len(stats['response_times']) > 10:
        stats['response_times'] = stats['response_times'][-10:]

def reset_conversation_stats():
    """Reset conversation statistics"""
    st.session_state.conversation_stats = {
        'start_time': None,
        'total_user_messages': 0,
        'total_assistant_messages': 0,
        'total_characters': 0,
        'estimated_tokens': 0,
        'response_times': []
    }

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from uploaded TXT file"""
    try:
        # Try to decode as UTF-8, fallback to latin-1
        try:
            text = txt_file.read().decode('utf-8')
        except UnicodeDecodeError:
            txt_file.seek(0)  # Reset file pointer
            text = txt_file.read().decode('latin-1')
        return text.strip()
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text"""
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return None

def update_document_context():
    """Update the document context for the conversation"""
    if st.session_state.uploaded_documents:
        context_parts = []
        for doc in st.session_state.uploaded_documents:
            context_parts.append(f"Document: {doc['name']}\n{doc['content']}")
        st.session_state.document_context = "\n\n".join(context_parts)
    else:
        st.session_state.document_context = ""

def initialize_conversation_chain():
    """Initialize the LangChain ConversationChain with ChatGroq"""
    try:
        # Get API key from environment variable
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            st.error("❌ GROQ_API_KEY environment variable not found. Please set your Groq API key.")
            st.stop()

        # Initialize ChatGroq with llama-3.1-8b-instant model
        llm = ChatGroq(
            api_key=SecretStr(groq_api_key),
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )

        # Initialize conversation memory
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")

        # Get the selected system prompt
        system_prompt = SYSTEM_PROMPTS.get(st.session_state.selected_prompt_style, SYSTEM_PROMPTS["Assistant"])

        # Add document context to system prompt if available
        if st.session_state.document_context:
            enhanced_prompt = f"{system_prompt}\n\nAdditional Context from Uploaded Documents:\n{st.session_state.document_context}\n\nPlease use this context to provide more informed and relevant responses when appropriate."
        else:
            enhanced_prompt = system_prompt

        # Create chat prompt template with system message
        prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Create conversation chain with system prompt
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )

        return conversation_chain

    except Exception as e:
        st.error(f"❌ Error initializing conversation chain: {str(e)}")
        return None

def export_conversation_json():
    """Export conversation history as JSON"""
    if not st.session_state.messages:
        return None

    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "model": "llama-3.1-8b-instant",
        "provider": "Groq API",
        "total_messages": len(st.session_state.messages),
        "conversation": st.session_state.messages
    }

    return json.dumps(export_data, indent=2)

def export_conversation_txt():
    """Export conversation history as plain text"""
    if not st.session_state.messages:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt_content = f"Conversation Export\n"
    txt_content += f"Exported on: {timestamp}\n"
    txt_content += f"Model: llama-3.1-8b-instant (Groq API)\n"
    txt_content += f"Total messages: {len(st.session_state.messages)}\n"
    txt_content += "=" * 50 + "\n\n"

    for i, message in enumerate(st.session_state.messages, 1):
        role = "You" if message["role"] == "user" else "Assistant"
        txt_content += f"{role}: {message['content']}\n\n"

    return txt_content

def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

def main():
    """Main application function"""

    # App header
    st.title("🤖 Context-Aware Chatbot")
    st.markdown("*Powered by LangChain, Groq API, and Llama-3.1-8b-Instant*")

    # Initialize conversation chain if not already done
    if st.session_state.conversation_chain is None:
        with st.spinner("Initializing chatbot..."):
            st.session_state.conversation_chain = initialize_conversation_chain()

    # Check if conversation chain was successfully initialized
    if st.session_state.conversation_chain is None:
        st.warning("⚠️ Chatbot initialization failed. Please check your API key and try again.")
        return

    # Sidebar with information and controls
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This is a context-aware chatbot that maintains conversation memory using LangChain's ConversationChain.")

        st.header("🔧 Model Info")
        st.write("**Model:** llama-3.1-8b-instant")
        st.write("**Provider:** Groq API")
        st.write("**Memory:** Conversation Buffer")

        # System Prompt Selection
        st.header("🎭 Conversation Style")
        current_style = st.session_state.selected_prompt_style

        new_style = st.selectbox(
            "Choose conversation style:",
            options=list(SYSTEM_PROMPTS.keys()),
            index=list(SYSTEM_PROMPTS.keys()).index(current_style),
            help="Different styles change how the AI responds and behaves"
        )

        # Show current style description
        st.info(f"**{new_style}:** {SYSTEM_PROMPTS[new_style][:100]}...")

        # Handle style change
        if new_style != current_style:
            st.session_state.selected_prompt_style = new_style
            # Reset both conversation chain and messages for new style
            st.session_state.conversation_chain = None
            st.session_state.messages = []
            st.session_state.show_clear_confirm = False  # Reset any pending clear confirmation
            # Reset conversation statistics
            reset_conversation_stats()
            st.success(f"🔄 Conversation style changed to '{new_style}'. Starting fresh conversation...")
            st.rerun()

        # Export functionality
        if st.session_state.messages:
            st.header("📤 Export Chat")

            # Generate filenames with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JSON export
            json_data = export_conversation_json()
            if json_data:
                st.download_button(
                    label="📄 Download as JSON",
                    data=json_data,
                    file_name=f"chat_export_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )

            # TXT export
            txt_data = export_conversation_txt()
            if txt_data:
                st.download_button(
                    label="📝 Download as Text",
                    data=txt_data,
                    file_name=f"chat_export_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # Clear conversation functionality with confirmation
        if st.session_state.messages:
            st.header("🗑️ Reset Chat")

            if not st.session_state.show_clear_confirm:
                if st.button("Clear Conversation", type="secondary", use_container_width=True):
                    st.session_state.show_clear_confirm = True
                    st.rerun()
            else:
                st.warning("⚠️ This will permanently delete your entire conversation history!")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Clear", type="primary", use_container_width=True):
                        st.session_state.messages = []
                        # Reset the conversation memory
                        if st.session_state.conversation_chain:
                            st.session_state.conversation_chain.memory.clear()
                        # Reset conversation statistics
                        reset_conversation_stats()
                        st.session_state.show_clear_confirm = False
                        st.success("Conversation cleared successfully!")
                        st.rerun()

                with col2:
                    if st.button("❌ Cancel", type="secondary", use_container_width=True):
                        st.session_state.show_clear_confirm = False
                        st.rerun()

        # Display conversation analytics
        if st.session_state.messages:
            st.header("📊 Analytics")

            stats = st.session_state.conversation_stats
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])

            # Basic message stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", f"{user_messages}")
                st.metric("Characters", f"{stats['total_characters']:,}")
            with col2:
                st.metric("Est. Tokens", f"{stats['estimated_tokens']:,}")
                if stats['response_times']:
                    avg_time = sum(stats['response_times']) / len(stats['response_times'])
                    st.metric("Avg Response", f"{avg_time:.1f}s")

            # Conversation duration
            if stats['start_time']:
                duration = datetime.now() - stats['start_time']
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                if hours > 0:
                    duration_str = f"{hours}h {minutes}m"
                elif minutes > 0:
                    duration_str = f"{minutes}m {seconds}s"
                else:
                    duration_str = f"{seconds}s"

                st.metric("Duration", duration_str)

            # Response time chart (if enough data)
            if len(stats['response_times']) >= 3:
                st.subheader("Response Times")
                st.line_chart(stats['response_times'][-10:])  # Last 10 response times

    # File Upload Section
    st.header("📎 Document Upload")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents to provide context for the conversation",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files that the AI can reference during the conversation"
    )

    # Process uploaded files
    if uploaded_files:
        new_documents = []
        for uploaded_file in uploaded_files:
            # Check if file is already uploaded
            existing_names = [doc['name'] for doc in st.session_state.uploaded_documents]
            if uploaded_file.name not in existing_names:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    content = process_uploaded_file(uploaded_file)
                    if content:
                        new_documents.append({
                            'name': uploaded_file.name,
                            'content': content[:5000],  # Limit content to 5000 chars to avoid token limits
                            'size': len(content),
                            'type': uploaded_file.type
                        })

        # Add new documents to session state
        if new_documents:
            st.session_state.uploaded_documents.extend(new_documents)
            update_document_context()
            # Reset conversation chain to include new context
            st.session_state.conversation_chain = None
            st.success(f"Successfully uploaded {len(new_documents)} document(s)!")
            st.rerun()

    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.subheader("📄 Uploaded Documents")

        for i, doc in enumerate(st.session_state.uploaded_documents):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**{doc['name']}** ({doc['type']})")
                st.caption(f"Size: {doc['size']:,} characters (showing first 5000)")

            with col2:
                # Show preview button
                if st.button("👁️ Preview", key=f"preview_{i}"):
                    st.text_area(
                        f"Preview of {doc['name']}",
                        doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                        height=150,
                        disabled=True,
                        key=f"preview_content_{i}"
                    )

            with col3:
                # Remove document button
                if st.button("🗑️ Remove", key=f"remove_{i}", type="secondary"):
                    st.session_state.uploaded_documents.pop(i)
                    update_document_context()
                    # Reset conversation chain to update context
                    st.session_state.conversation_chain = None
                    st.rerun()

        if st.button("🗑️ Remove All Documents", type="secondary"):
            st.session_state.uploaded_documents = []
            update_document_context()
            st.session_state.conversation_chain = None
            st.success("All documents removed!")
            st.rerun()

    # Main chat interface
    st.header("💬 Chat")

    # Display existing chat history
    display_chat_history()

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response using conversation chain
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Track response time
                    start_time = datetime.now()

                    # Get response from conversation chain
                    response = st.session_state.conversation_chain.predict(input=user_input)

                    # Calculate response time
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds()

                    # Display response
                    st.write(response)

                    # Add assistant response to session state
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Update conversation statistics
                    update_conversation_stats(user_input, response, response_time)

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.write("Please try again or check your API connection.")

    # Display helpful information at the bottom
    if not st.session_state.messages:
        st.info("👋 Welcome! To begin, upload a document using the file uploader above and ask any question about its content.")

if __name__ == "__main__":
    main()



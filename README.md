# High-Speed RAG Chatbot with Groq & LangChain

A high-performance, context-aware chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on user-uploaded documents. Built with Python and Streamlit, it leverages the blazing-fast Groq API for real-time inference and LangChain for robust conversational memory management.

<img width="1908" height="897" alt="image" src="https://github.com/user-attachments/assets/fdc7a3a2-7aac-478c-affa-c9b13b643511" />


---

## About The Project

This project was developed to create a highly responsive and intelligent document analysis tool. Traditional chatbots often suffer from latency, making the user experience clunky. By integrating the Groq API, which powers LLMs on specialized LPUs (Language Processing Units), this application achieves near-instantaneous response times, making the conversation feel natural and fluid.

The core functionality is its RAG pipeline, which allows the AI to ingest information from PDF and TXT files and use that knowledge base to provide accurate, context-specific answers.

## Key Features

* **Retrieval-Augmented Generation (RAG):** Upload PDF or TXT documents and ask questions directly about their content.
* **High-Speed Inference:** Powered by the Groq API for a real-time, lag-free conversational experience.
* **Conversational Memory:** The chatbot remembers previous turns in the conversation for natural, context-aware follow-up questions, thanks to LangChain's `ConversationChain`.
* **Switchable AI Personas:** Choose from multiple conversation styles (e.g., Assistant, Teacher, Scientist) to change the AI's personality and response format.
* **Customizable UI:** Features a sleek, user-friendly interface with a toggle for Light and Dark modes.
* **Conversation Analytics:** The sidebar displays real-time statistics, including message counts, estimated token usage, and average response time.
* **Export Functionality:** Download the complete chat history as a JSON or plain text file.

## Tech Stack

* **Backend:** Python
* **Frontend:** Streamlit
* **AI Framework:** LangChain
* **LLM Provider:** Groq API (running Llama 3.1)
* **PDF Processing:** PyPDF2

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.9 or higher
* A free API key from [GroqCloud](https://console.groq.com/keys)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    * Create a file named `.env` in the root of your project directory.
    * Add your Groq API key to this file in the following format:
        ```
        GROQ_API_KEY="your_actual_api_key_goes_here"
        ```
    *(Note: The `app.py` script is configured to read this environment variable, but Streamlit Community Cloud requires setting this in the advanced settings as a Secret.)*

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will now be running on your local machine.

## License

Distributed under the MIT License. See `LICENSE` for more information.

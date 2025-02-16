
🔍 **Multimedia Retrieval-Augmented Generation (RAG) Framework for Unified Knowledge Extraction**

This application is a Streamlit-based interface that allows users to upload and process multiple data sources—including PDFs, web URLs, YouTube videos, and images—for retrieval-augmented question answering using OpenAI's GPT models. It leverages the LangChain library for document loading, splitting, embedding, and retrieval, and utilizes OpenAI's API for language model interactions.

## Features

- **Data Sources**: Upload PDFs, input web URLs, YouTube video links, and image URLs.
- **Data Processing**: Extracts and processes text from the provided sources.
- **Embeddings**: Generates embeddings using OpenAI's embedding models.
- **Retrieval**: Stores embeddings in a FAISS vector store for efficient similarity search.
- **Question Answering**: Allows users to ask questions; the application provides answers based on retrieved context.
- **Chat History**: Maintains a history of the conversation with the assistant.

## Requirements

- Python 3.9 or higher
- OpenAI API key

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruvlunawath/Multi-media-Rag.git
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

**Using `requirements.txt`**

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

**Without `requirements.txt`**

If `requirements.txt` is not available, install the required packages manually:

```bash
pip install streamlit langchain openai chromadb pypdf unstructured youtube-transcript-api httpx python-dotenv
```

## Setup

### 1. OpenAI API Key

The application requires an OpenAI API key to function. You can provide the API key in one of two ways:

- **Option A**: Create a `.env` file in the project directory and add your OpenAI API key.

  ```env
  OPENAI_API_KEY=your-openai-api-key
  ```

- **Option B**: Enter your OpenAI API key directly into the application sidebar when running.

### 2. Update the Script Filename

Ensure that the script filename matches when running the application. If the script is named differently (e.g., `app.py`), adjust the run command accordingly.

## Usage

### 1. Run the Application

```bash
streamlit run app.py
```

Replace `app.py` with the name of your Python script if it's different.

### 2. Application Interface

- **Configuration** (Sidebar):

  - **OpenAI API Key**: Enter your API key if you haven't set up the `.env` file.
  - **Choose Model**: Select the OpenAI GPT model to use (e.g., `gpt-3.5-turbo`, `gpt-4`, `gpt-4o-mini`).

- **Data Sources** (Sidebar):

  - **Upload PDFs**: Click to upload one or more PDF files.
  - **Enter Web URLs**: Input web URLs, separated by commas.
  - **Enter YouTube URLs**: Input YouTube video URLs, separated by commas.
  - **Enter Image URLs**: Input image URLs, separated by commas.
  - **Load Data**: Click the **📥 Load Data** button to process and index the data.

- **Chat Interface**:

  - **Ask a Question**: Input your question in the text field.
  - **Submit**: Press Enter or click outside the text field to submit your question.
  - **Responses**: The assistant will provide answers based on the uploaded data.
  - **Chat History**: View previous questions and answers in the interface.

## Example Workflow

1. **Upload Data**:

   - Upload PDF files containing relevant documents.
   - Enter web URLs pointing to articles or resources.
   - Input YouTube video URLs for transcription and inclusion.
   - Provide image URLs (note: image processing is limited).

2. **Load Data**:

   - Click on the **📥 Load Data** button.
   - Wait for the application to process and index the data.

3. **Ask Questions**:

   - Enter your question in the **Ask a question** field.
   - Example questions:
     - "What are the key takeaways from the uploaded PDFs?"
     - "Summarize the main points from the web articles."
     - "What topics are covered in the YouTube videos?"

4. **Receive Answers**:

   - The assistant will generate responses based on the retrieved context.
   - Review the answers directly in the chat interface.

## Notes

- **Image Processing Limitation**: OpenAI's GPT models accessed via the API cannot process images directly. The application includes placeholder functionality for images.

- **Error Handling**: The application includes error handling for invalid URLs and data processing issues. Errors are displayed in the interface.

## Dependencies

- **Streamlit**: For the web interface.
- **LangChain**: For handling language model interactions and data processing.
- **OpenAI**: To access OpenAI's GPT models.
- **FAISS**: For efficient similarity search in vector embeddings.
- **YouTube Transcript API**: To retrieve transcripts from YouTube videos.
- **PyMuPDF**: To extract text from PDFs.
- **Unstructured**: To process HTML and text data from web pages.
- **HTTPX**: For HTTP requests.
- **Python Dotenv**: To manage environment variables.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Push your branch to your forked repository.
5. Open a pull request describing your changes.



## Acknowledgments

- Thanks to the developers of [LangChain](https://github.com/hwchase17/langchain) for providing powerful tools for language model applications.
- Thanks to the [OpenAI](https://openai.com/) team for their GPT models.

## Contact

For questions or support, please open an issue in the repository.


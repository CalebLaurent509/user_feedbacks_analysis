# User Feedback Analysis System

An intelligent feedback analysis system that leverages advanced Natural Language Processing, Retrieval-Augmented Generation (RAG), and intent classification to automatically process, categorize, and respond to user feedback. The system dynamically discovers emerging intents, provides contextual responses using company documents, and sends automated alerts for new trending feedback patterns.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [API Documentation](#api-documentation)
8. [Intent Classification](#intent-classification)
9. [RAG System](#rag-system)
10. [Alert System](#alert-system)
11. [Data Management](#data-management)
12. [Development](#development)
13. [Contributing](#contributing)
14. [License](#license)
15. [Contact](#contact)

## Features

- **Intelligent Intent Classification**: Automatically categorizes user feedback using transformer-based models
- **Dynamic Intent Discovery**: Discovers and promotes emerging feedback patterns automatically
- **RAG-Powered Responses**: Generates contextual responses using company knowledge base
- **Real-time Processing**: FastAPI-based REST API for real-time feedback processing
- **Automated Alerts**: Email notifications for new trending intents and feedback patterns
- **Persistent Data Management**: Comprehensive data storage and retrieval system
- **Scalable Architecture**: Modular design supporting easy extension and scaling
- **Multi-modal Document Support**: Processes PDF and text documents for knowledge base
- **Advanced Analytics**: Statistical tracking and summarization of feedback trends

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Intent         │    │   RAG System    │
│   Web Service   │ ── │   Classification │ ── │   & Response    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data          │    │   Alert          │    │   Document      │
│   Management    │    │   System         │    │   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **MeaningEngine**: Advanced intent classification and emerging pattern detection
2. **CompanyRAGSystem**: Retrieval-augmented generation for contextual responses
3. **AlertSender**: Automated notification system for feedback trends
4. **DataManager**: Persistent storage and retrieval of feedback data

## Project Structure

```
user_feedbacks_analysis/
├── core/
│   ├── __init__.py
│   ├── logic.py           # Intent classification and meaning engine
│   ├── rag_system.py      # RAG implementation and document processing
│   ├── alert_sender.py    # Email alert system
│   └── execute.py         # Additional execution utilities
├── data/
│   ├── data_manager.py    # Data persistence and management
│   ├── base_intentions.json # Base intent categories
│   ├── intent_stats.json  # Intent occurrence statistics
│   ├── intent_feedbacks.json # Historical feedback data
│   ├── company_docs/      # Company knowledge base documents
│   └── company_chroma_db/ # Vector database for RAG
├── models/
│   └── model.py          # LLM and embedding model configuration
├── notebooks/            # Jupyter notebooks for analysis
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (create from .env.example)
├── Dockerfile          # Container configuration
├── CHANGELOG.md        # Version history
└── README.md           # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key for GPT-3.5-turbo
- SMTP credentials for email alerts (optional)
- 4GB+ RAM for optimal performance
- Internet connection for model downloads

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/CalebLaurent509/user_feedbacks_analysis.git
   cd user_feedbacks_analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Prepare company documents**
   ```bash
   # Add your company documents to data/company_docs/
   mkdir -p data/company_docs
   # Copy PDF and text files to this directory
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Email Alert Configuration (Optional)
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
TO_EMAIL=admin@company.com

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

### Model Configuration

The system uses the following models by default:
- **LLM**: GPT-3.5-turbo for intent simplification and summarization
- **Embeddings**: sentence-transformers/paraphrase-MiniLM-L6-v2
- **Intent Classification**: facebook/bart-large-mnli

Models are automatically downloaded on first use.

## Usage

### Starting the Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Basic API Usage

```python
import requests

# Process feedback
response = requests.post(
    "http://localhost:8000/process-feedback",
    json={
        "feedback": "The checkout process is confusing and takes too long",
        "admin_email": "admin@company.com"
    }
)

result = response.json()
print(f"Intent: {result['intent_label']}")
print(f"Response: {result['rag_response']}")
```

### Command Line Usage

```bash
# Process a single feedback
curl -X POST "http://localhost:8000/process-feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "feedback": "Your product quality has improved significantly",
    "admin_email": "admin@company.com"
  }'
```

## API Documentation

### Endpoints

#### POST /process-feedback

Processes user feedback and returns intent classification and RAG response.

**Request Body:**
```json
{
  "feedback": "User feedback text",
  "admin_email": "admin@company.com"
}
```

**Response:**
```json
{
  "results": {
    "question": "Processed question",
    "result": "RAG response",
    "source_documents": ["Document sources"]
  },
  "intent_label": "ClassifiedIntent",
  "user_input": "Original feedback",
  "rag_response": "Generated response"
}
```

**Response Codes:**
- `200`: Successful processing
- `400`: Invalid request format
- `500`: Internal server error

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Intent Classification

### How It Works

1. **Embedding Generation**: Converts feedback text to semantic embeddings
2. **Similarity Matching**: Compares against known intent categories
3. **Emerging Intent Detection**: Identifies new patterns not in existing categories
4. **Dynamic Promotion**: Promotes frequently occurring patterns to official intents

### Intent Lifecycle

```
New Feedback → Similarity Check → Known Intent? 
                                     ↓ No
                              Emerging Intent → Count Threshold? 
                                                    ↓ Yes
                                              Promote to Official Intent → Send Alert
```

### Customizing Intent Categories

Edit `data/base_intentions.json` to modify the base intent categories:

```json
[
  "Product Quality Feedback",
  "Service Experience",
  "Technical Issues",
  "Feature Requests",
  "Billing Inquiries",
  "General Satisfaction"
]
```

## RAG System

### Document Processing

The RAG system processes company documents to provide contextual responses:

1. **Document Loading**: Supports PDF and text files
2. **Text Chunking**: Splits documents into manageable chunks
3. **Vector Storage**: Creates embeddings using ChromaDB
4. **Retrieval**: Finds relevant context for user queries
5. **Generation**: Produces responses using retrieved context

### Adding Company Documents

```bash
# Add documents to the knowledge base
cp your_document.pdf data/company_docs/
cp policy_document.txt data/company_docs/

# Restart the application to process new documents
```

### RAG Configuration

Customize the RAG system in `core/rag_system.py`:

```python
# Chunk configuration
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks

# Retrieval configuration
k_documents = 5        # Number of documents to retrieve
```

## Alert System

### Email Notifications

The system sends automated email alerts when:
- New intents are promoted from emerging patterns
- Feedback volume exceeds thresholds
- System errors occur

### Configuring Alerts

1. **SMTP Setup**: Configure email credentials in `.env`
2. **Alert Rules**: Modify thresholds in `core/logic.py`
3. **Templates**: Customize email templates in `core/alert_sender.py`

### Alert Types

- **Intent Promotion**: New trending feedback patterns
- **Volume Alerts**: High feedback volume periods
- **System Health**: Error notifications and performance alerts

## Data Management

### Data Storage

The system uses JSON files for persistent storage:

- `intent_stats.json`: Intent occurrence counters
- `intent_feedbacks.json`: Historical feedback data
- `base_intentions.json`: Official intent categories

### Data Operations

```python
from data.data_manager import *

# Save intent statistics
save_intent_counts(intent_count_dict)

# Load feedback history
feedback_data = {}
load_intent_feedbacks(feedback_data)

# Export data for analysis
save_intent_feedbacks(processed_feedbacks)
```

### Backup and Recovery

```bash
# Backup data files
cp data/*.json backup/

# Restore from backup
cp backup/*.json data/
```

## Development

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black isort flake8

# Run tests
pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test categories
pytest tests/test_intent_classification.py
pytest tests/test_rag_system.py
```

### Docker Deployment

```bash
# Build image
docker build -t user-feedback-analysis .

# Run container
docker run -p 8000:8000 --env-file .env user-feedback-analysis
```

### Performance Optimization

1. **Vector Database**: Consider FAISS for large-scale deployments
2. **Caching**: Implement Redis for embedding caching
3. **Async Processing**: Use Celery for background tasks
4. **Model Optimization**: Use quantized models for faster inference

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Follow code standards** (run `black` and `isort`)
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit changes** (`git commit -m 'Add amazing feature'`)
7. **Push to branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints where possible
- Document all public functions and classes
- Maintain test coverage above 80%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Caleb Laurent**
- Email: contact@caleb-dev.info
- GitHub: [@CalebLaurent509](https://github.com/CalebLaurent509)
- Project Link: [https://github.com/CalebLaurent509/user_feedbacks_analysis](https://github.com/CalebLaurent509/user_feedbacks_analysis)

## Acknowledgments

- OpenAI for GPT-3.5-turbo language model
- Hugging Face for transformer models and libraries
- LangChain community for RAG framework
- ChromaDB for vector database capabilities
- FastAPI team for the excellent web framework

---

*Transforming feedback into insights, one message at a time.*

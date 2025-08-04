"""
Setup configuration for User Feedback Analysis System
"""

import os
from setuptools import setup, find_packages

readme_path = os.path.join(os.path.dirname(__file__), "README.md")

with open(readme_path, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="user-feedback-analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.10",
        "chromadb>=0.4.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
        "pypdf>=3.17.0",
        "secure-smtplib>=0.1.1",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "loguru>=0.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0"
        ],
        "performance": [
            "faiss-cpu>=1.7.4"
        ]
    },
    entry_points={
        "console_scripts": [
            "feedback-analysis=main:app",
        ],
    },
    author="Caleb Laurent",
    author_email="laurentcaleb99@gmail.com",
    description="Intelligent user feedback analysis system with RAG and intent classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CalebLaurent509/user_feedbacks_analysis",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Office/Business :: Customer Relationship Management",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
    ],
    keywords="nlp, feedback-analysis, intent-classification, rag, customer-service, ai, machine-learning",
    python_requires=">=3.8",
    project_urls={
        "Bug Reports": "https://github.com/CalebLaurent509/user_feedbacks_analysis/issues",
        "Source": "https://github.com/CalebLaurent509/user_feedbacks_analysis",
        "Documentation": "https://github.com/CalebLaurent509/user_feedbacks_analysis/blob/main/README.md",
        "Changelog": "https://github.com/CalebLaurent509/user_feedbacks_analysis/blob/main/CHANGELOG.md",
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt", "*.yml", "*.yaml"],
    },
)

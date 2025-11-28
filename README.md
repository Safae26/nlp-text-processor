# NLP Text Processor

A powerful web application for natural language text analysis and preprocessing. This tool enables users to input text and apply various NLP techniques including tokenization, stop word removal, stemming, lemmatization, and part-of-speech tagging.

## Features

- **Text Tokenization**: Split text into words and sentences using multiple methods (NLTK, spaCy, TextBlob, split)
- **Text Normalization**: 
  - Lowercasing
  - Contraction correction
  - Punctuation removal
  - Multiple spaces removal
  - Spelling correction
  - Emoji to text conversion
- **Advanced Analysis**:
  - Stop words removal
  - POS tagging (Part-of-Speech)
  - Stemming
  - Lemmatization
- **Real-time Processing**: Interactive interface with progress tracking
- **Statistical Summary**: Detailed analysis metrics

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Safae26/nlp-text-processor.git
   cd nlp-text-processor
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter your text in the input box.
2. Select the desired NLP techniques.
3. Click the "Process Text" button to see the results.
4. View the processed text and analysis metrics in the main panel.

## Dependencies

- Python 3.7+
- Streamlit
- NLTK
- spaCy
- TextBlob
- emoji
- contractions

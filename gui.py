import streamlit as st
import pandas as pd
from nltk.stem import SnowballStemmer
import spacy
import nltk
from app import *

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Streamlit configuration
st.set_page_config(
    page_title="NLP Text Processor",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Main header - CENTERED TITLE
st.markdown("<h1 style='text-align: center;'>NLP Text Processor</h1>", unsafe_allow_html=True)

# Main text area
text_input = st.text_area(
    "Enter your text to analyze:",
    height=200,
    key="text_input",
    help="Paste or type the text you want to preprocess and analyze",
    placeholder="Example: Natural Language Processing (NLP) is fascinating. It enables machines to understand and process human language..."
)

# Processing parameters
st.markdown("### Processing Parameters")

# Create tabs to organize parameters
tab1, tab2, tab3 = st.tabs(["Tokenization", "Normalization", "Advanced Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        word_tokenizer_type = st.selectbox(
            "Word Tokenizer:",
            ("nltk", "split", "spaCy", "TextBlob"),
            index=0,
            key="word_tokenizer_type",
            help="Method to split text into individual words"
        )
    with col2:
        sentence_tokenizer_type = st.selectbox(
            "Sentence Tokenizer:",
            ("nltk", "split", "spaCy", "TextBlob"),
            index=0,
            key="sentence_tokenizer_type",
            help="Method to split text into sentences"
        )

with tab2:
    normalization_operations = st.multiselect(
        "Cleaning Operations:",
        ("Lowercasing", "Contraction Correction", "Punctuation Removal",
         "Multiple Spaces Removal", "Spelling Correction",
         "Emoji to Text Conversion"),
        key="normalization_operations",
        help="Text preprocessing and cleaning steps"
    )

with tab3:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        remove_stop_words_flag = st.checkbox(
            "Remove Stop Words",
            value=False,
            key="remove_stop_words",
            help="Remove common words without meaning (the, a, an, in, etc.)"
        )
    with col2:
        pos_tagging = st.checkbox(
            "POS Tagging",
            value=False,
            key="pos_tagging",
            help="Identify grammatical nature of each word"
        )
    with col3:
        stemming = st.checkbox(
            "Stemming",
            value=False,
            key="stemming",
            help="Reduce words to their root form"
        )
    with col4:
        lemmatization = st.checkbox(
            "Lemmatization",
            value=False,
            key="lemmatization",
            help="Reduce words to their canonical form"
        )

# Process button placed after all parameters
st.markdown("---")
process_button = st.button(
    "Start Analysis",
    key="process_button",
    use_container_width=True,
    type="primary"
)

# Results section
if process_button and text_input:
    st.markdown("### Analysis Results")
    
    # Progress bar for interactivity
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    sentences = sentence_tokenizer(text_input, sentence_tokenizer_type)
    
    for i, sent in enumerate(sentences):
        # Update progress bar
        progress = (i + 1) / len(sentences)
        progress_bar.progress(progress)
        status_text.text(f"Processing sentence {i+1}/{len(sentences)}...")
        
        # Card for each sentence
        with st.expander(f"Sentence {i+1}: {sent[:80]}..." if len(sent) > 80 else f"Sentence {i+1}: {sent}", expanded=False):
            st.markdown(f"**Original text:** `{sent}`")
            
            # Initialize variables
            normalized_text = sent
            tokens = []
            pos_tags = []
            stemmed_tokens = []
            lemmatized_tokens = []

            # Normalization
            if normalization_operations:
                for operation in normalization_operations:
                    normalized_text = normalize(normalized_text, operation)
                st.markdown(f"**Normalized text:** `{normalized_text}`")

            # Tokenization
            tokens = word_tokenizer(normalized_text if normalization_operations else sent, word_tokenizer_type)
            
            # Stop words removal
            if remove_stop_words_flag:
                original_count = len(tokens)
                tokens = remove_stop_words(tokens)
                removed_count = original_count - len(tokens)
                st.markdown(f"**Stop words removed:** {removed_count} word(s)")
            
            # Prepare data for DataFrame
            data = {'Token': tokens}
            
            # POS tagging
            if pos_tagging and tokens:
                pos_tags = POS_tag(tokens)
                data['POS'] = [tag[1] for tag in pos_tags]
            
            # Stemming
            if stemming and tokens:
                stemmer = SnowballStemmer('english')  # Changé à anglais
                stemmed_tokens = [stemmer.stem(token) for token in tokens]
                data['Stem'] = stemmed_tokens
            
            # Lemmatization
            if lemmatization and tokens:
                try:
                    nlp = spacy.load("en_core_web_sm")  # Changé à modèle anglais
                    doc = nlp(" ".join(tokens))
                    lemmatized_tokens = [token.lemma_ for token in doc]
                    data['Lemma'] = lemmatized_tokens
                except OSError:
                    st.warning("English spaCy model not installed. Installation: `python -m spacy download en_core_web_sm`")
                    lemmatization = False

            # Create and display DataFrame
            if tokens:
                # Ensure all columns have the same length
                max_length = len(tokens)
                for key in data.keys():
                    if len(data[key]) < max_length:
                        # Fill with empty values if needed
                        data[key] = data[key] + [''] * (max_length - len(data[key]))
                    elif len(data[key]) > max_length:
                        # Truncate if needed
                        data[key] = data[key][:max_length]
                
                df = pd.DataFrame(data)
                
                # Conditional column display
                display_columns = ['Token']
                if pos_tagging and 'POS' in df.columns:
                    display_columns.append('POS')
                if stemming and 'Stem' in df.columns:
                    display_columns.append('Stem')
                if lemmatization and 'Lemma' in df.columns:
                    display_columns.append('Lemma')
                
                st.markdown(f"**Analyzed tokens ({len(tokens)}):**")
                st.dataframe(df[display_columns], use_container_width=True)
            else:
                st.info("No tokens to display after processing.")
    
    # Clean up progress bar
    progress_bar.empty()
    status_text.empty()
            
    # Analysis summary
    st.markdown("### Statistical Summary")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Sentences analyzed", len(sentences))
    
    with col_stats2:
        total_tokens = sum(len(word_tokenizer(sent, word_tokenizer_type)) for sent in sentences)
        st.metric("Total tokens", total_tokens)
    
    with col_stats3:
        avg_tokens = total_tokens / len(sentences) if sentences else 0
        st.metric("Average tokens/sentence", f"{avg_tokens:.1f}")

elif process_button and not text_input:
    st.warning("Please enter text to analyze first!")

# Footer
st.markdown("---")
st.markdown("**Made by Safae** | © 2025")
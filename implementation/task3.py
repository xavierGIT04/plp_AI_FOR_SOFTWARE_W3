# ==============================================
# Task 3: NLP with spaCy (Adapted for test.txt)
# Goal: NER for product/brand, Rule-based Sentiment
# ==============================================

import spacy
import re
import os  # To check if the file exists

print("--- Starting Task 3: spaCy NLP (Adapted for test.txt) ---")


# --- FILE LOADING AND PREPROCESSING ---

def load_reviews_from_fasttext(file_path="test.txt"):
    """
    Loads data from a fastText-like format file (e.g., __label__2 review text).
    Returns a list of review strings.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please ensure the file is in the correct directory.")
        return []

    reviews = []
    # Regex to capture the label and the rest of the line (the review text)
    label_pattern = re.compile(r"^__label__\d\s+")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove the label prefix (e.g., '__label__2 ') to get clean text
            review_text = label_pattern.sub('', line).strip()
            if review_text:
                reviews.append(review_text)

    print(f"Successfully loaded {len(reviews)} reviews from {file_path}.")
    return reviews


# Load the large English model (better for NER)
# Note: You may need to run: python -m spacy download en_core_web_lg
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("\n!! IMPORTANT !! Large spaCy model 'en_core_web_lg' not found.")
    print("Please run: 'python -m spacy download en_core_web_lg' in your terminal.")
    # Fallback to a smaller model for code execution
    nlp = spacy.load("en_core_web_sm")
    print("Using 'en_core_web_sm' as a fallback, which may yield lower NER accuracy.")

# Load your data
reviews = load_reviews_from_fasttext()
if not reviews:
    exit()  # Stop if no data was loaded


# --- Goal 1: Named Entity Recognition (NER) ---
def extract_entities(text):
    """Performs NER and filters for relevant product/brand labels."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # Filter for entities related to products/brands/companies
        # ORG (Organization/Company), PRODUCT, and sometimes PERSON/GPE for brand names
        if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE"]:
            entities.append((ent.text, ent.label_))
    return entities


# --- Goal 2: Rule-Based Sentiment Analysis ---
# Using a slightly richer set of keywords for a more robust analysis
def analyze_sentiment(text):
    """Simple rule-based sentiment by counting keywords."""
    positive_words = ["great", "best", "love", "awesome", "wonderful", "splendid",
                      "amazing", "fun", "excellent", "well built", "thrilled", "happy", "sweet"]
    negative_words = ["died", "disappointing", "terrible", "bad", "poor", "crapped out",
                      "useless", "bust", "boring", "waste of money", "brok", "not as expected"]

    text_lower = text.lower()

    # Use sum of counts for simple scoring
    pos_score = sum(text_lower.count(word) for word in positive_words)
    neg_score = sum(text_lower.count(word) for word in negative_words)

    if pos_score > neg_score * 1.5:  # Require a strong positive bias
        return "Positive"
    elif neg_score > pos_score * 1.5:  # Require a strong negative bias
        return "Negative"
    else:
        return "Neutral/Mixed"


# --- Deliverable: Code Snippet and Output ---
print("\n--- Processing First 5 Extracted Reviews ---")

# Process only the first 5 for the deliverable snippet
for i, review in enumerate(reviews[:5]):
    # Use spaCy's sentence segmentation for cleaner presentation
    review_summary = " ".join([sent.text for sent in nlp(review).sents][:2])
    print(f"\nReview {i + 1} Text: \"{review_summary[:100]}...\"")

    # 1. NER Extraction
    entities = extract_entities(review)
    # Filter entities list to keep only unique ones for cleaner output
    unique_entities = sorted(list(set(entities)))

    product_brands = ", ".join([f"'{text}' ({label})" for text, label in unique_entities])
    if not product_brands:
        product_brands = "None Detected"
    print(f"  Extracted Entities (NER): {product_brands}")

    # 2. Sentiment Analysis
    sentiment = analyze_sentiment(review)
    print(f"  Sentiment (Rule-Based): {sentiment}")

print("\n--- Task 3 Complete ---")
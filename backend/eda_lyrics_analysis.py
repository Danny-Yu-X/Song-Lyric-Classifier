import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from better_profanity import profanity
import warnings

from nltk.tokenize import word_tokenize #might use later
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  #might use later
from sklearn.decomposition import LatentDirichletAllocation  #might use later
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback to old punkt if punkt_tab fails
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load profanity checker
profanity.load_censor_words()

# Configuration
# Use smaller dataset for faster testing, or larger for full analysis
USE_SMALLER_DATASET = False  # Set to True for faster testing
DATA_FILE_LARGE = '../data/finalCombinedPlaylist.csv'  # ~2,700 songs
DATA_FILE_SMALL = ''  # ~1,300 songs
DATA_FILE = DATA_FILE_SMALL if USE_SMALLER_DATASET else DATA_FILE_LARGE
OUTPUT_DIR = 'eda_outputs'
FIG_SIZE = (12, 8)

def load_data(file_path):
    """Load the dataset from CSV file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Loaded {len(df)} songs")
    
    # Convert AgeAppropriate to numeric, handling any invalid values
    print(f"\nAgeAppropriate value counts (before cleaning):")
    print(df['AgeAppropriate'].value_counts())
    
    # Convert to numeric, coercing errors to NaN
    df['AgeAppropriate'] = pd.to_numeric(df['AgeAppropriate'], errors='coerce')
    
    # Remove rows with NaN (invalid AgeAppropriate values)
    invalid_count = df['AgeAppropriate'].isna().sum()
    if invalid_count > 0:
        print(f"Removing {invalid_count} rows with invalid AgeAppropriate values...")
        df = df.dropna(subset=['AgeAppropriate']).copy()
    
    # Convert to int
    df['AgeAppropriate'] = df['AgeAppropriate'].astype(int)
    
    print(f"\nAgeAppropriate distribution (after cleaning):")
    print(df['AgeAppropriate'].value_counts().sort_index())
    
    return df

def preprocess_text(text):
    """Basic text preprocessing."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_text_features(df):
    """Extract various text-based features from lyrics."""
    print("\nExtracting text features...")
    print(f"  Processing {len(df)} songs...")
    
    # Create features DataFrame with same index as df
    features = pd.DataFrame(index=df.index)
    
    # Basic statistics
    features['word_count'] = df['Lyrics'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    features['char_count'] = df['Lyrics'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    features['char_count_no_spaces'] = df['Lyrics'].apply(
        lambda x: len(str(x).replace(' ', '')) if pd.notna(x) else 0
    )
    features['avg_word_length'] = features['char_count_no_spaces'] / (features['word_count'] + 1)
    features['sentence_count'] = df['Lyrics'].apply(
        lambda x: len(re.split(r'[.!?]+', str(x))) if pd.notna(x) else 0
    )
    
    # Vocabulary richness metrics
    features['unique_words'] = df['Lyrics'].apply(
        lambda x: len(set(str(x).lower().split())) if pd.notna(x) else 0
    )
    features['vocab_richness'] = features['unique_words'] / (features['word_count'] + 1)
    
    # Sentiment analysis
    print("  Computing sentiment scores...")
    print("    (This may take a few minutes for large datasets...)")
    total = len(df)
    sentiments = []
    for idx, lyrics in enumerate(df['Lyrics']):
        if (idx + 1) % 500 == 0:
            print(f"    Progress: {idx + 1}/{total} songs processed...")
        if pd.notna(lyrics):
            sentiment = TextBlob(str(lyrics)).sentiment
        else:
            sentiment = TextBlob("").sentiment
        sentiments.append(sentiment)
    
    features['polarity'] = [s.polarity for s in sentiments]
    features['subjectivity'] = [s.subjectivity for s in sentiments]
    print("    [OK] Sentiment analysis complete!")
    
    # Profanity detection
    print("  Checking for profanity...")
    print("    (This may take a few minutes for large datasets...)")
    total = len(df)
    profanity_results = []
    for idx, lyrics in enumerate(df['Lyrics']):
        if (idx + 1) % 500 == 0:
            print(f"    Progress: {idx + 1}/{total} songs processed...")
        if pd.isna(lyrics):
            profanity_results.append((0, 0))
        else:
            text = str(lyrics)
            has_prof = 1 if profanity.contains_profanity(text) else 0
            prof_count = has_prof  # Simplified for performance
            profanity_results.append((has_prof, prof_count))
    
    features['has_profanity'] = [x[0] for x in profanity_results]
    features['profanity_count'] = [x[1] for x in profanity_results]
    print("    [OK] Profanity check complete!")
    
    # Special characters and patterns
    print("  Computing special character metrics...")
    features['exclamation_count'] = df['Lyrics'].apply(
        lambda x: str(x).count('!') if pd.notna(x) else 0
    )
    features['question_count'] = df['Lyrics'].apply(
        lambda x: str(x).count('?') if pd.notna(x) else 0
    )
    features['uppercase_ratio'] = df['Lyrics'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1) if pd.notna(x) else 0
    )
    
    # Repetition metrics
    print("  Computing repetition scores...")
    features['repetition_score'] = df['Lyrics'].apply(
        lambda x: calculate_repetition_score(str(x)) if pd.notna(x) else 0
    )
    
    print("  [OK] All features extracted!")
    return features

def calculate_repetition_score(text):
    """Calculate how repetitive the text is."""
    words = text.lower().split()
    if len(words) < 2:
        return 0
    word_counts = Counter(words)
    # Ratio of most common word to total words
    if len(word_counts) > 0:
        most_common_count = word_counts.most_common(1)[0][1]
        return most_common_count / len(words)
    return 0

def analyze_word_frequencies(df, top_n=30):
    """Analyze word frequencies for kids vs adults songs."""
    print("\nAnalyzing word frequencies...")
    print("  Tokenizing and processing lyrics...")
    
    kids_lyrics = ' '.join(df[df['AgeAppropriate'] == 1]['Lyrics'].astype(str))
    adults_lyrics = ' '.join(df[df['AgeAppropriate'] == 0]['Lyrics'].astype(str))
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    def get_top_words(text, n):
        # Use simple split instead of word_tokenize to avoid NLTK dependency issues
        # This is sufficient for word frequency analysis
        words = text.lower().split()
        # Filter: only alphabetic words, not stopwords, length > 2
        words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation).isalpha() 
                and w.strip(string.punctuation) not in stop_words 
                and len(w.strip(string.punctuation)) > 2]
        word_freq = Counter(words)
        return dict(word_freq.most_common(n))
    
    kids_words = get_top_words(kids_lyrics, top_n)
    adults_words = get_top_words(adults_lyrics, top_n)
    
    return kids_words, adults_words

def create_visualizations(df, features):
    """Create all visualizations for EDA."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating Visualizations...")
    print("="*60)
    
    # 1. Class Distribution
    print("\n1. Class Distribution")
    plt.figure(figsize=FIG_SIZE)
    class_counts = df['AgeAppropriate'].value_counts().sort_index()
    # Ensure correct order: 0 (Adults) first, then 1 (Kids)
    adults_count = class_counts.get(0, 0)
    kids_count = class_counts.get(1, 0)
    plt.bar(['Adults (0)', 'Kids (1)'], [adults_count, kids_count], color=['#e74c3c', '#3498db'])
    plt.title('Distribution of Songs by Age Appropriateness', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Songs', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.text(0, adults_count + 10, str(adults_count), ha='center', fontsize=12, fontweight='bold')
    plt.text(1, kids_count + 10, str(kids_count), ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Word Count Distribution
    print("2. Word Count Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (label, data) in enumerate([('Adults', df[df['AgeAppropriate'] == 0]),
                                          ('Kids', df[df['AgeAppropriate'] == 1])]):
        word_counts = features.loc[data.index, 'word_count']
        axes[idx].hist(word_counts, bins=50, color=['#e74c3c', '#3498db'][idx], alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{label} Songs - Word Count Distribution', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Word Count', fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].axvline(word_counts.mean(), color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {word_counts.mean():.0f}')
        axes[idx].legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Comparison Bar Chart
    print("3. Feature Comparison")
    comparison_features = ['word_count', 'char_count', 'unique_words', 'vocab_richness', 
                          'polarity', 'subjectivity', 'repetition_score']
    kids_features = features[df['AgeAppropriate'] == 1][comparison_features].mean()
    adults_features = features[df['AgeAppropriate'] == 0][comparison_features].mean()
    
    x = np.arange(len(comparison_features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars1 = ax.bar(x - width/2, kids_features.values, width, label='Kids', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, adults_features.values, width, label='Adults', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Average Value', fontsize=12)
    ax.set_title('Average Feature Values: Kids vs Adults Songs', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in comparison_features], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sentiment Analysis
    print("4. Sentiment Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (label, data_idx) in enumerate([('Adults', df[df['AgeAppropriate'] == 0].index),
                                            ('Kids', df[df['AgeAppropriate'] == 1].index)]):
        polarity = features.loc[data_idx, 'polarity']
        subjectivity = features.loc[data_idx, 'subjectivity']
        
        scatter = axes[idx].scatter(polarity, subjectivity, alpha=0.5, 
                                    color=['#e74c3c', '#3498db'][idx], s=50)
        axes[idx].set_xlabel('Polarity (Negative to Positive)', fontsize=12)
        axes[idx].set_ylabel('Subjectivity (Objective to Subjective)', fontsize=12)
        axes[idx].set_title(f'{label} Songs - Sentiment Analysis', fontsize=14, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        axes[idx].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Profanity Analysis
    print("5. Profanity Analysis")
    profanity_stats = pd.DataFrame({
        'Category': ['Kids', 'Adults'],
        'With Profanity': [
            features[df['AgeAppropriate'] == 1]['has_profanity'].sum(),
            features[df['AgeAppropriate'] == 0]['has_profanity'].sum()
        ],
        'Without Profanity': [
            (df['AgeAppropriate'] == 1).sum() - features[df['AgeAppropriate'] == 1]['has_profanity'].sum(),
            (df['AgeAppropriate'] == 0).sum() - features[df['AgeAppropriate'] == 0]['has_profanity'].sum()
        ]
    })
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(profanity_stats))
    width = 0.35
    ax.bar(x - width/2, profanity_stats['With Profanity'], width, label='With Profanity', 
           color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, profanity_stats['Without Profanity'], width, label='Without Profanity', 
           color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Number of Songs', fontsize=12)
    ax.set_title('Profanity Detection in Songs', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(profanity_stats['Category'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_profanity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Top Words Comparison
    print("6. Top Words Comparison")
    kids_words, adults_words = analyze_word_frequencies(df, top_n=20)
    
    # Get common words for comparison
    all_kids_words = set(kids_words.keys())
    all_adults_words = set(adults_words.keys())
    common_words = all_kids_words & all_adults_words
    kids_only = all_kids_words - all_adults_words
    adults_only = all_adults_words - all_kids_words
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Kids-only words
    if kids_only:
        kids_only_data = {w: kids_words[w] for w in list(kids_only)[:15]}
        axes[0].barh(range(len(kids_only_data)), list(kids_only_data.values()), color='#3498db')
        axes[0].set_yticks(range(len(kids_only_data)))
        axes[0].set_yticklabels(list(kids_only_data.keys()))
        axes[0].set_title('Top Words - Kids Only', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Frequency')
        axes[0].invert_yaxis()
    
    # Common words comparison
    if common_words:
        common_list = list(common_words)[:15]
        kids_freq = [kids_words.get(w, 0) for w in common_list]
        adults_freq = [adults_words.get(w, 0) for w in common_list]
        x = np.arange(len(common_list))
        width = 0.35
        axes[1].barh(x - width/2, kids_freq, width, label='Kids', color='#3498db', alpha=0.8)
        axes[1].barh(x + width/2, adults_freq, width, label='Adults', color='#e74c3c', alpha=0.8)
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(common_list)
        axes[1].set_title('Top Common Words Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Frequency')
        axes[1].legend()
        axes[1].invert_yaxis()
    
    # Adults-only words
    if adults_only:
        adults_only_data = {w: adults_words[w] for w in list(adults_only)[:15]}
        axes[2].barh(range(len(adults_only_data)), list(adults_only_data.values()), color='#e74c3c')
        axes[2].set_yticks(range(len(adults_only_data)))
        axes[2].set_yticklabels(list(adults_only_data.keys()))
        axes[2].set_title('Top Words - Adults Only', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Frequency')
        axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_top_words_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Correlation Heatmap
    print("7. Feature Correlation Heatmap")
    feature_cols = ['word_count', 'char_count', 'unique_words', 'vocab_richness', 
                   'polarity', 'subjectivity', 'repetition_score', 'has_profanity']
    correlation_matrix = features[feature_cols].corr()
    
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Word Clouds
    print("8. Word Clouds")
    kids_lyrics = ' '.join(df[df['AgeAppropriate'] == 1]['Lyrics'].astype(str))
    adults_lyrics = ' '.join(df[df['AgeAppropriate'] == 0]['Lyrics'].astype(str))
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for idx, (lyrics, label, color) in enumerate([(kids_lyrics, 'Kids', 'Blues'),
                                                   (adults_lyrics, 'Adults', 'Reds')]):
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap=color, max_words=100).generate(lyrics)
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f'{label} Songs - Word Cloud', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_word_clouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Feature Distribution by Class
    print("9. Feature Distribution by Class")
    feature_to_plot = 'vocab_richness'
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    kids_data = features[df['AgeAppropriate'] == 1][feature_to_plot]
    adults_data = features[df['AgeAppropriate'] == 0][feature_to_plot]
    
    ax.hist(kids_data, bins=50, alpha=0.6, label='Kids', color='#3498db', edgecolor='black')
    ax.hist(adults_data, bins=50, alpha=0.6, label='Adults', color='#e74c3c', edgecolor='black')
    ax.set_xlabel(feature_to_plot.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {feature_to_plot.replace("_", " ").title()} by Class', 
                fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_vocab_richness_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Statistical Summary Table
    print("10. Statistical Summary")
    summary_stats = features.groupby(df['AgeAppropriate']).agg({
        'word_count': ['mean', 'std', 'median'],
        'vocab_richness': ['mean', 'std', 'median'],
        'polarity': ['mean', 'std', 'median'],
        'subjectivity': ['mean', 'std', 'median'],
        'has_profanity': 'sum'
    })
    
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    print(summary_stats)
    
    # Save summary to file
    with open(f'{OUTPUT_DIR}/statistical_summary.txt', 'w') as f:
        f.write("STATISTICAL SUMMARY OF FEATURES\n")
        f.write("="*60 + "\n\n")
        f.write(str(summary_stats))
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("FEATURE IMPORTANCE FOR ML MODEL\n")
        f.write("="*60 + "\n\n")
        f.write("Based on the analysis, the following features show significant differences:\n\n")
        f.write("1. Profanity Detection: Strong indicator (adults songs have more profanity)\n")
        f.write("2. Vocabulary Richness: Kids songs may have simpler vocabulary\n")
        f.write("3. Sentiment (Polarity/Subjectivity): Different emotional patterns\n")
        f.write("4. Word Count: Different song lengths\n")
        f.write("5. Repetition Score: Kids songs may be more repetitive\n")
        f.write("6. Unique Words: Vocabulary diversity\n")
    
    print(f"\nAll visualizations saved to '{OUTPUT_DIR}/' directory")
    print("Statistical summary saved to 'statistical_summary.txt'")

def main():
    """Main function to run the EDA."""
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS - SPOTIFY LYRICS CLASSIFIER")
    print("="*60)
    
    # Load data
    df = load_data(DATA_FILE)
    
    # Extract features
    features = extract_text_features(df)
    
    # Create visualizations
    create_visualizations(df, features)
    
    # Save features for potential ML use
    features_with_labels = features.copy()
    features_with_labels['AgeAppropriate'] = df['AgeAppropriate'].values
    features_with_labels.to_csv(f'{OUTPUT_DIR}/extracted_features.csv', index=False)
    print(f"\nExtracted features saved to '{OUTPUT_DIR}/extracted_features.csv'")
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()


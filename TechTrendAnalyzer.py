import pandas as pd
import nltk
import os
from datetime import datetime
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TechNewsAnalyzer:
    def __init__(self, csv_path="phoronix.csv"):
        """Initialize the Technology News Analysis System"""
        self.csv_path = csv_path
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Technology categories for classification
        self.tech_categories = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 
                     'ai', 'ml', 'algorithm', 'automation'],
            'Hardware': ['cpu', 'gpu', 'processor', 'hardware', 'intel', 'amd', 'nvidia'],
            'Software': ['software', 'application', 'program', 'code', 'development'],
            'Security': ['security', 'hack', 'breach', 'malware', 'ransomware', 'encryption'],
            'Cloud': ['cloud', 'aws', 'azure', 'google cloud', 'saas', 'paas', 'iaas'],
            'Mobile': ['mobile', 'smartphone', 'ios', 'android', 'app', 'mobile app'],
            'Gaming': ['game', 'gaming', 'console', 'playstation', 'xbox', 'nintendo']
        }
        
        self.load_data()
    
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Successfully loaded {len(self.df)} articles from {self.csv_path}")
            
            # Convert date column to datetime
            if 'created_at' in self.df.columns:
                self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
            
            # Fill missing values
            self.df['title'] = self.df['title'].fillna('')
            self.df['text'] = self.df['text'].fillna('')
            
            print("Data preparation completed.")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text or pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and special patterns
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Lemmatization
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(processed_tokens)
    
    def detect_category(self, text: str) -> str:
        """Detect the primary technology category"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.tech_categories.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'General'
    
    def analyze_trends(self, num_clusters: int = 5):
        """Analyze trends and create visualizations"""
        if self.df.empty:
            print("No data available for analysis")
            return
        
        # Process all titles
        processed_titles = [self.preprocess_text(title) for title in self.df['title']]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(processed_titles)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Add categories and clusters to dataframe
        self.df['category'] = self.df['title'].apply(self.detect_category)
        self.df['cluster'] = cluster_labels
        
        # Create timestamp for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = f"analysis_results_{timestamp}"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Category Distribution Plot
        plt.figure(figsize=(12, 6))
        category_counts = self.df['category'].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Distribution of Technology Categories')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'category_distribution.png'))
        plt.close()
        
        # 2. Cluster Analysis Plot
        plt.figure(figsize=(12, 6))
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
        plt.title('Distribution of Article Clusters')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'cluster_distribution.png'))
        plt.close()
        
        # Save analysis results
        results_file = os.path.join(analysis_dir, 'trend_analysis.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Technology News Trend Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Category Analysis
            f.write("Category Distribution:\n")
            f.write("-" * 20 + "\n")
            for category, count in category_counts.items():
                f.write(f"{category}: {count} articles\n")
            f.write("\n")
            
            # Cluster Analysis
            f.write("Cluster Analysis:\n")
            f.write("-" * 20 + "\n")
            for cluster_id in range(num_clusters):
                cluster_articles = self.df[self.df['cluster'] == cluster_id]
                f.write(f"\nCluster {cluster_id} ({len(cluster_articles)} articles):\n")
                
                # Get top titles for this cluster
                top_titles = cluster_articles['title'].head(5)
                for i, title in enumerate(top_titles, 1):
                    f.write(f"{i}. {title}\n")
                
                # Get category distribution within cluster
                cluster_categories = cluster_articles['category'].value_counts()
                f.write("\nCategory distribution in this cluster:\n")
                for category, count in cluster_categories.items():
                    f.write(f"  - {category}: {count} articles\n")
        
        print(f"\nAnalysis results have been saved in directory: {analysis_dir}")

def main():
    """Run the technology news analysis"""
    print(" Initializing Technology News Analysis System...")
    print(" Analyzing trends from Phoronix dataset")
    print()
    
    analyzer = TechNewsAnalyzer()
    analyzer.analyze_trends()
    
    print("\nAnalysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 
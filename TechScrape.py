#1. Moch. Udhay Yunussabil	                    163221004
#2. Valiantino Ramandhika A.                    163221038
#3. Patrick Andrasena Tumengkol                 163221077

import pandas as pd
import string
import nltk
import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import time
import re
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import random
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TechNewsIRSystem:
    def __init__(self, db_path="tech_news_ir.db"):
        """Initialize the Real-Time Technology News IR System"""
        self.db_path = db_path
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        # Technology-specific keywords for topic detection
        self.tech_topics = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 
                     'ai', 'ml', 'chatgpt', 'openai', 'GGUF', 'LLM'],
            'Blockchain': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'crypto', 'defi', 
                          'nft', 'web3',],
            'Cloud Computing': ['cloud', 'aws', 'azure', 'google cloud', 'saas', 'paas', 'iaas', 
                               'serverless', 'kubernetes'],
            'Cybersecurity': ['cybersecurity', 'security', 'hack', 'breach', 'malware', 'ransomware', 
                             'phishing', 'encryption', 'attack'],
            'Mobile Tech': ['mobile', 'smartphone', 'ios', 'android', 'app', 'mobile app', 'tablet'],
            'IoT': ['iot', 'internet of things', 'smart home', 'connected device', 'sensor'],
            'Data Science': ['data science', 'big data', 'analytics', 'data mining', 'visualization'],
            'Programming': ['python', 'javascript', 'java', 'programming', 'software development', 
                           'coding', 'framework'],
        }
        
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for technology news"""
        if not text or pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, and special patterns
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and numbers but keep some tech symbols
        text = re.sub(r'[^\w\s#@\-\.]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Lemmatization
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(processed_tokens)
    
    def get_user_agent(self) -> str:
        """Return a random user agent to avoid blocking"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        return random.choice(user_agents)
    
    def fetch_rss_feed(self, rss_url: str, source_name: str) -> List[Dict]:
        """Fetch articles from RSS feed"""
        articles = []
        try:
            # Set user agent to avoid blocking
            headers = {'User-Agent': self.get_user_agent()}
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries:
                # Parse published date
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6])
                    else:
                        published_date = datetime.now()
                except:
                    published_date = datetime.now()
                
                # Get content (try different fields)
                content = ""
                if hasattr(entry, 'content') and entry.content:
                    content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                elif hasattr(entry, 'description'):
                    content = entry.description
                
                # Clean HTML from content
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text().strip()
                
                article = {
                    'title': entry.title if hasattr(entry, 'title') else 'No Title',
                    'content': content,
                    'url': entry.link if hasattr(entry, 'link') else '',
                    'published_date': published_date,
                    'source': source_name
                }
                articles.append(article)
                
        except Exception as e:
            print(f"Error fetching RSS from {rss_url}: {e}")
        
        return articles
    
    def scrape_article_content(self, url: str) -> str:
        """Scrape full article content from URL"""
        try:
            headers = {'User-Agent': self.get_user_agent()}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                
                # Try different content selectors for different sites
                content_selectors = [
                    'article',
                    '.article-content',
                    '.post-content',
                    '.entry-content',
                    '.content',
                    'main',
                    '[role="main"]'
                ]
                
                content = ""
                for selector in content_selectors:
                    content_element = soup.select_one(selector)
                    if content_element:
                        # Get text from paragraphs
                        paragraphs = content_element.find_all('p')
                        if paragraphs:
                            content = ' '.join([p.get_text().strip() for p in paragraphs])
                            break
                
                # Fallback: get all paragraph text
                if not content:
                    all_paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text().strip() for p in all_paragraphs])
                
                return content[:5000]  # Limit content length
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        
        return ""
    
    def fetch_tech_news(self, query: str, num_articles: int = 20) -> List[Dict]:
        """
        Fetch technology articles from Google Scholar
        """
        print("Fetching articles from Google Scholar...")
        
        # Fetch from Google Scholar
        articles = self.fetch_google_scholar(query, num_articles)
        
        # Sort by published date
        articles.sort(key=lambda x: x['published_date'], reverse=True)
        return articles[:num_articles]

    def fetch_google_scholar(self, query: str, num_articles: int) -> List[Dict]:
        """Fetch articles from Google Scholar"""
        articles = []
        try:
            headers = {
                'User-Agent': self.get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://scholar.google.com/'
            }
            
            # Construct search URL
            search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&hl=en&as_sdt=0,5"
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all article entries
                entries = soup.find_all('div', class_='gs_ri')
                
                for entry in entries[:num_articles]:
                    title_elem = entry.find('h3', class_='gs_rt')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    url = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    
                    # Get publication info
                    pub_info = entry.find('div', class_='gs_a')
                    pub_text = pub_info.get_text() if pub_info else ''
                    
                    # Extract year from publication info
                    year_match = re.search(r'\b(19|20)\d{2}\b', pub_text)
                    year = int(year_match.group()) if year_match else datetime.now().year
                    
                    article = {
                        'title': title,
                        'content': pub_text,  # Using publication info as content
                        'url': url,
                        'published_date': datetime(year, 1, 1),  # Using year only
                        'source': 'Google Scholar'
                    }
                    articles.append(article)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(2)
                    
        except Exception as e:
            print(f"Error fetching from Google Scholar: {e}")
        
        return articles

    def detect_topic_category(self, text: str) -> str:
        """Detect the primary technology topic category using weighted keyword scores."""
        text_lower = text.lower()
        topic_scores = defaultdict(float)

        for topic, keywords in self.tech_topics.items():
            score = 0
            for keyword in keywords:
                # Weight by the square of the number of words in the keyword
                # This gives much higher scores to specific multi-word phrases
                weight = len(keyword.split()) ** 2
                
                # Use regex with \b for word boundaries to ensure whole word/phrase matching
                # and prevent partial matches (e.g., 'ai' in 'train').
                # We use count of matches multiplied by weight.
                matches = re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower)
                if matches:
                    score += len(matches) * weight
            
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        return 'General Tech'
    
    def calculate_sentiment_score(self, text: str) -> float:
        """Simple sentiment analysis for news articles"""
        positive_words = ['breakthrough', 'revolutionary', 'innovative', 'advanced', 'successful', 
                         'improved', 'efficient', 'powerful', 'amazing', 'excellent']
        negative_words = ['vulnerable', 'threat', 'attack', 'breach', 'failure', 'problem', 
                         'issue', 'concern', 'risk', 'dangerous']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (pos_count - neg_count) / total_words
        return max(-1.0, min(1.0, sentiment))  # Normalize to [-1, 1]
    
    def expand_query(self, query: str, articles: List[Dict]) -> Tuple[str, List[str]]:
        """Intelligent query expansion based on retrieved articles"""
        if not articles:
            return query, []
        
        # Combine all article content
        all_text = ' '.join([self.preprocess_text(article['title'] + ' ' + article['content']) 
                            for article in articles])
        
        if not all_text.strip():
            return query, []
        
        # Extract key terms using TF-IDF
        try:
            tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform([all_text])
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = np.argsort(scores)[::-1][:10]
            expansion_terms = [feature_names[i] for i in top_indices 
                             if scores[i] > 0 and feature_names[i].lower() not in query.lower()]
            
            expanded_query = query + ' ' + ' '.join(expansion_terms[:5])
            return expanded_query, expansion_terms[:5]
        except:
            return query, []
    
    def track_title_categories(self, articles: List[Dict]) -> Dict[str, List[str]]:
        """Track and categorize article titles"""
        category_titles = defaultdict(list)
        
        for article in articles:
            title = article['title']
            category = article['topic_category']
            category_titles[category].append(title)
        
        return dict(category_titles)

    def analyze_title_trends(self, category_titles: Dict[str, List[str]]) -> Dict:
        """Analyze trends in titles by category"""
        analysis = {
            'category_counts': {},
            'common_terms': {},
            'title_lengths': {}
        }
        
        for category, titles in category_titles.items():
            # Count titles per category
            analysis['category_counts'][category] = len(titles)
            
            # Analyze common terms
            all_terms = []
            for title in titles:
                terms = self.preprocess_text(title).split()
                all_terms.extend(terms)
            
            term_freq = Counter(all_terms)
            analysis['common_terms'][category] = dict(term_freq.most_common(10))
            
            # Analyze title lengths
            lengths = [len(title.split()) for title in titles]
            analysis['title_lengths'][category] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'min': min(lengths),
                'max': max(lengths)
            }
        
        return analysis

    def create_category_analysis_table(self, all_category_titles: Dict[str, List[str]]) -> pd.DataFrame:
        """Create a consolidated analysis table for all categories"""
        analysis_data = []
        
        # Create directory for storing titles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        titles_dir = f"category_titles_{timestamp}"
        if not os.path.exists(titles_dir):
            os.makedirs(titles_dir)
        
        for category, titles in all_category_titles.items():
            # Calculate statistics
            title_count = len(titles)
            avg_length = np.mean([len(title.split()) for title in titles])
            median_length = np.median([len(title.split()) for title in titles])
            
            # Get common terms
            all_terms = []
            for title in titles:
                terms = self.preprocess_text(title).split()
                all_terms.extend(terms)
            
            term_freq = Counter(all_terms)
            top_terms = dict(term_freq.most_common(5))
            
            # Save titles to file
            category_filename = os.path.join(titles_dir, f"{category.lower().replace('/', '_')}_titles.txt")
            with open(category_filename, 'w', encoding='utf-8') as f:
                f.write(f"Category: {category}\n")
                f.write(f"Total Titles: {title_count}\n")
                f.write(f"Average Title Length: {avg_length:.2f}\n")
                f.write(f"Median Title Length: {median_length:.2f}\n")
                f.write("\nTop Terms:\n")
                for term, count in top_terms.items():
                    f.write(f"- {term}: {count}\n")
                f.write("\nTitles:\n")
                f.write("="*80 + "\n")
                for title in titles:
                    f.write(f"{title}\n")
            
            # Add to analysis data
            analysis_data.append({
                'Category': category,
                'Title Count': title_count,
                'Avg Title Length': round(avg_length, 2),
                'Median Title Length': round(median_length, 2),
                'Top Terms': ', '.join(f"{term}({count})" for term, count in top_terms.items())
            })
        
        # Create DataFrame
        df = pd.DataFrame(analysis_data)
        return df.sort_values('Title Count', ascending=False), titles_dir

    def create_category_visualizations(self, analysis_table: pd.DataFrame, titles_dir: str):
        """Create visualizations for category analysis"""
        # Create directory for visualizations
        viz_dir = os.path.join(titles_dir, "visualizations")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Title Count Distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(data=analysis_table, x='Category', y='Title Count')
        plt.title('Distribution of Titles by Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'title_count_distribution.png'))
        plt.close()
        
        # 2. Title Length Analysis
        plt.figure(figsize=(12, 6))
        sns.barplot(data=analysis_table, x='Category', y='Avg Title Length')
        plt.title('Average Title Length by Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'title_length_analysis.png'))
        plt.close()
        
        # 3. Title Length vs Count Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis_table['Avg Title Length'], analysis_table['Title Count'], 
                   s=analysis_table['Title Count']*2, alpha=0.6)
        
        # Add labels for each point
        for i, row in analysis_table.iterrows():
            plt.annotate(row['Category'], 
                        (row['Avg Title Length'], row['Title Count']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Relationship between Title Length and Count')
        plt.xlabel('Average Title Length')
        plt.ylabel('Number of Titles')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'title_length_vs_count.png'))
        plt.close()
        
        # 4. Category Distribution Pie Chart
        plt.figure(figsize=(10, 10))
        plt.pie(analysis_table['Title Count'], 
                labels=analysis_table['Category'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Category Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'category_distribution_pie.png'))
        plt.close()

    def create_additional_visualizations(self, csv_path: str, titles_dir: str):
        """Create additional visualizations from the CSV data"""
        # Read the CSV data
        df = pd.read_csv(csv_path)
        
        # Create directory for additional visualizations
        additional_viz_dir = os.path.join(titles_dir, "additional_visualizations")
        if not os.path.exists(additional_viz_dir):
            os.makedirs(additional_viz_dir)
        
        # 1. Title Count Distribution with Trend Line
        plt.figure(figsize=(15, 7))
        plt.bar(df['Category'], df['Title Count'], color='skyblue', alpha=0.7)
        plt.plot(df['Category'], df['Title Count'], 'r--', alpha=0.5)
        plt.title('Title Count Distribution with Trend Line')
        plt.xlabel('Category')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(additional_viz_dir, 'title_count_trend.png'))
        plt.close()
        
        # 2. Title Length Comparison
        plt.figure(figsize=(15, 7))
        x = np.arange(len(df['Category']))
        width = 0.35
        
        plt.bar(x - width/2, df['Avg Title Length'], width, label='Average Length', color='lightgreen')
        plt.bar(x + width/2, df['Median Title Length'], width, label='Median Length', color='lightblue')
        
        plt.xlabel('Category')
        plt.ylabel('Title Length')
        plt.title('Average vs Median Title Length by Category')
        plt.xticks(x, df['Category'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(additional_viz_dir, 'title_length_comparison.png'))
        plt.close()
        
        # 3. Category Distribution Heatmap
        plt.figure(figsize=(12, 8))
        # Create a correlation matrix of title counts and lengths
        corr_matrix = df[['Title Count', 'Avg Title Length', 'Median Title Length']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Title Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(additional_viz_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # 4. Title Length Distribution Box Plot
        plt.figure(figsize=(15, 7))
        data_to_plot = []
        labels = []
        for _, row in df.iterrows():
            # Create synthetic data based on mean and median
            data = np.random.normal(row['Avg Title Length'], 
                                  row['Avg Title Length'] * 0.2, 
                                  int(row['Title Count']))
            data_to_plot.append(data)
            labels.append(row['Category'])
        
        plt.boxplot(data_to_plot, labels=labels)
        plt.title('Title Length Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Title Length')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(additional_viz_dir, 'title_length_boxplot.png'))
        plt.close()
        
        # 5. Category Distribution with Percentage
        plt.figure(figsize=(12, 8))
        total_titles = df['Title Count'].sum()
        percentages = (df['Title Count'] / total_titles * 100).round(1)
        
        plt.pie(df['Title Count'], 
                labels=[f"{cat}\n({pct}%)" for cat, pct in zip(df['Category'], percentages)],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                explode=[0.05] * len(df))  # Slight separation between slices
        
        plt.title('Category Distribution with Percentages')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(additional_viz_dir, 'category_distribution_percentage.png'))
        plt.close()
        
        # 6. Title Length vs Count Scatter with Regression
        plt.figure(figsize=(12, 8))
        plt.scatter(df['Avg Title Length'], df['Title Count'], 
                   s=df['Title Count']*2, alpha=0.6, c='blue')
        
        # Add regression line
        z = np.polyfit(df['Avg Title Length'], df['Title Count'], 1)
        p = np.poly1d(z)
        plt.plot(df['Avg Title Length'], p(df['Avg Title Length']), 
                "r--", alpha=0.8)
        
        # Add labels for each point
        for i, row in df.iterrows():
            plt.annotate(row['Category'], 
                        (row['Avg Title Length'], row['Title Count']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Title Length vs Count with Regression Line')
        plt.xlabel('Average Title Length')
        plt.ylabel('Number of Titles')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(additional_viz_dir, 'title_length_vs_count_regression.png'))
        plt.close()

    def search_and_analyze(self, query: str, num_articles: int = 20, num_clusters: int = None) -> Dict:
        """Main search function with comprehensive analysis, now with LSA+KMeans clustering for categorization"""
        start_time = time.time()
        
        # Fetch articles
        articles = self.fetch_tech_news(query, num_articles)
        
        if not articles:
            return {'error': 'No articles found for the given query'}
        
        # Process articles
        processed_articles = []
        all_texts = []
        for article in articles:
            processed_content = self.preprocess_text(article['title'] + ' ' + article['content'])
            topic_category = self.detect_topic_category(article['title'] + ' ' + article['content'])
            sentiment_score = self.calculate_sentiment_score(article['title'] + ' ' + article['content'])
            processed_article = {
                **article,
                'processed_content': processed_content,
                'topic_category': topic_category,
                'sentiment_score': sentiment_score
            }
            processed_articles.append(processed_article)
            all_texts.append(processed_content)
        
        # Set default number of clusters to number of main tech topics
        if num_clusters is None:
            num_clusters = len(self.tech_topics)
        
        # LSA + KMeans clustering on processed article texts
        cluster_labels = None
        if all_texts and any(text.strip() for text in all_texts):
            try:
                tfidf_matrix = self.vectorizer_tfidf.fit_transform(all_texts)
                # LSA dimensionality reduction
                n_components = min(100, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
                lsa = TruncatedSVD(n_components=n_components, random_state=42)
                lsa_matrix = lsa.fit_transform(tfidf_matrix)
                # KMeans clustering on LSA-reduced features
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(lsa_matrix)
                # Assign cluster label to each article
                for i, article in enumerate(processed_articles):
                    article['cluster_label'] = int(cluster_labels[i])
            except Exception as e:
                print(f"Error in LSA+KMeans clustering: {e}")
                for article in processed_articles:
                    article['cluster_label'] = -1
        else:
            for article in processed_articles:
                article['cluster_label'] = -1
        
        # Track and analyze titles by category (original, keyword-based)
        category_titles = self.track_title_categories(processed_articles)
        title_analysis = self.analyze_title_trends(category_titles)
        
        # Calculate TF-IDF and relevance scores
        if all_texts and any(text.strip() for text in all_texts):
            try:
                tfidf_matrix = self.vectorizer_tfidf.fit_transform(all_texts)
                # Process query
                processed_query = self.preprocess_text(query)
                query_vector = self.vectorizer_tfidf.transform([processed_query])
                # Calculate cosine similarity
                similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
                for i, article in enumerate(processed_articles):
                    article['relevance_score'] = float(similarities[i])
                # Sort by relevance
                processed_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            except Exception as e:
                print(f"Error in TF-IDF calculation: {e}")
                for article in processed_articles:
                    article['relevance_score'] = 0.0
        
        # Query expansion
        expanded_query, expansion_terms = self.expand_query(query, articles)
        
        processing_time = time.time() - start_time
        
        # Generate analysis
        analysis = self.generate_analysis(processed_articles)
        analysis['title_analysis'] = title_analysis
        
        # Also group articles by cluster for possible downstream use
        cluster_titles = defaultdict(list)
        for article in processed_articles:
            cluster = article.get('cluster_label', -1)
            cluster_titles[cluster].append(article['title'])
        
        return {
            'query': query,
            'expanded_query': expanded_query,
            'expansion_terms': expansion_terms,
            'num_results': len(processed_articles),
            'processing_time': processing_time,
            'articles': processed_articles,
            'analysis': analysis,
            'category_titles': category_titles,
            'cluster_titles': dict(cluster_titles),
            'num_clusters': num_clusters
        }
    
    def generate_analysis(self, articles: List[Dict]) -> Dict:
        """Generate comprehensive analysis of retrieved articles"""
        if not articles:
            return {}
        
        # Topic distribution
        topic_counts = Counter([article['topic_category'] for article in articles])
        
        # Sentiment analysis
        sentiments = [article.get('sentiment_score', 0.0) for article in articles]
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        # Source distribution
        source_counts = Counter([article['source'] for article in articles])
        
        # Time distribution
        recent_count = sum(1 for article in articles 
                          if (datetime.now() - article['published_date']).days < 1)
        
        # Relevance statistics
        relevance_scores = [article.get('relevance_score', 0.0) for article in articles]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        return {
            'topic_distribution': dict(topic_counts),
            'average_sentiment': avg_sentiment,
            'source_distribution': dict(source_counts),
            'recent_articles_count': recent_count,
            'average_relevance': avg_relevance,
            'total_articles': len(articles)
        }
    
    def get_trending_topics(self) -> List[Dict]:
        """Get trending topics from recent articles"""
        print("Analyzing trending topics...")
        
        # Fetch recent articles without specific query
        recent_articles = self.fetch_tech_news("", num_articles=50)
        
        if not recent_articles:
            return []
        
        # Process all article content
        all_text = []
        for article in recent_articles:
            processed_content = self.preprocess_text(article['title'] + ' ' + article['content'])
            if processed_content.strip():
                all_text.append(processed_content)
        
        if not all_text:
            return []
        
        try:
            # Use TF-IDF to find trending terms
            tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 3), min_df=2)
            tfidf_matrix = tfidf.fit_transform(all_text)
            feature_names = tfidf.get_feature_names_out()
            
            # Get average TF-IDF scores across all documents
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top trending terms
            top_indices = np.argsort(mean_scores)[::-1][:20]
            trending_topics = []
            
            for idx in top_indices:
                if mean_scores[idx] > 0:
                    term = feature_names[idx]
                    topic_category = self.detect_topic_category(term)
                    trending_topics.append({
                        'term': term,
                        'score': float(mean_scores[idx]),
                        'category': topic_category
                    })
            
            return trending_topics
            
        except Exception as e:
            print(f"Error analyzing trending topics: {e}")
            return []
    
    def visualize_results(self, search_results: Dict):
        """Create visualizations for search results and save them to files"""
        if 'articles' not in search_results or not search_results['articles']:
            print("No articles to visualize")
            return
        
        articles = search_results['articles']
        analysis = search_results['analysis']
        query = search_results['query']
        category_titles = search_results.get('category_titles', {})
        
        # Create timestamp and directory for this runtime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runtime_dir = f"analysis_results_{timestamp}"
        
        # Create directory if it doesn't exist
        if not os.path.exists(runtime_dir):
            os.makedirs(runtime_dir)
        
        base_filename = f"analysis_{query.replace(' ', '_')}"
        
        # 1. Topic Distribution Plot
        if analysis.get('topic_distribution'):
            plt.figure(figsize=(10, 6))
            topics = list(analysis['topic_distribution'].keys())
            counts = list(analysis['topic_distribution'].values())
            plt.bar(topics, counts, color='skyblue')
            plt.title(f'Topic Distribution for Query: "{query}"')
            plt.xlabel('Topics')
            plt.ylabel('Number of Articles')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(runtime_dir, f"{base_filename}_topic_distribution.png"))
            plt.close()
        
        # 2. Title Analysis Visualizations
        if category_titles:
            # 2.1 Title Length Distribution by Category
            plt.figure(figsize=(12, 6))
            title_lengths = []
            categories = []
            for category, titles in category_titles.items():
                lengths = [len(title.split()) for title in titles]
                title_lengths.extend(lengths)
                categories.extend([category] * len(lengths))
            
            sns.boxplot(x=categories, y=title_lengths)
            plt.title(f'Title Length Distribution by Category for Query: "{query}"')
            plt.xlabel('Category')
            plt.ylabel('Number of Words in Title')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(runtime_dir, f"{base_filename}_title_lengths.png"))
            plt.close()
            
            # 2.2 Common Terms by Category
            if analysis.get('title_analysis', {}).get('common_terms'):
                plt.figure(figsize=(15, 8))
                common_terms = analysis['title_analysis']['common_terms']
                
                # Create a heatmap of term frequencies
                terms = set()
                for category_terms in common_terms.values():
                    terms.update(category_terms.keys())
                
                term_matrix = np.zeros((len(common_terms), len(terms)))
                term_list = list(terms)
                
                for i, (category, terms_dict) in enumerate(common_terms.items()):
                    for j, term in enumerate(term_list):
                        term_matrix[i, j] = terms_dict.get(term, 0)
                
                sns.heatmap(term_matrix, 
                          xticklabels=term_list,
                          yticklabels=list(common_terms.keys()),
                          cmap='YlOrRd')
                plt.title(f'Common Terms by Category for Query: "{query}"')
                plt.xlabel('Terms')
                plt.ylabel('Category')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(runtime_dir, f"{base_filename}_common_terms.png"))
                plt.close()
        
        # 3. Sentiment Distribution Plot
        plt.figure(figsize=(10, 6))
        sentiments = [article.get('sentiment_score', 0.0) for article in articles]
        plt.hist(sentiments, bins=10, color='lightgreen', alpha=0.7)
        plt.title(f'Sentiment Score Distribution for Query: "{query}"')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(runtime_dir, f"{base_filename}_sentiment_distribution.png"))
        plt.close()
        
        # 4. Source Distribution Plot
        if analysis.get('source_distribution'):
            plt.figure(figsize=(10, 6))
            sources = list(analysis['source_distribution'].keys())
            source_counts = list(analysis['source_distribution'].values())
            plt.pie(source_counts, labels=sources, autopct='%1.1f%%')
            plt.title(f'Source Distribution for Query: "{query}"')
            plt.tight_layout()
            plt.savefig(os.path.join(runtime_dir, f"{base_filename}_source_distribution.png"))
            plt.close()
        
        # Save analysis summary and article details to a text file
        summary_file = os.path.join(runtime_dir, f"{base_filename}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Analysis Summary for Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Articles: {len(articles)}\n")
            f.write(f"Average Relevance: {analysis.get('average_relevance', 0):.3f}\n")
            f.write(f"Average Sentiment: {analysis.get('average_sentiment', 0):.3f}\n")
            f.write(f"Recent Articles (24h): {analysis.get('recent_articles_count', 0)}\n")
            f.write(f"Topics Found: {list(analysis.get('topic_distribution', {}).keys())}\n\n")
            
            # Add title analysis
            if analysis.get('title_analysis'):
                f.write("="*80 + "\n")
                f.write("TITLE ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                for category, stats in analysis['title_analysis']['category_counts'].items():
                    f.write(f"Category: {category}\n")
                    f.write(f"Number of Titles: {stats}\n")
                    f.write(f"Common Terms: {dict(analysis['title_analysis']['common_terms'][category])}\n")
                    f.write(f"Title Length Statistics:\n")
                    length_stats = analysis['title_analysis']['title_lengths'][category]
                    f.write(f"  Mean: {length_stats['mean']:.1f}\n")
                    f.write(f"  Median: {length_stats['median']:.1f}\n")
                    f.write(f"  Min: {length_stats['min']}\n")
                    f.write(f"  Max: {length_stats['max']}\n")
                    f.write("\n")
            
            # Add article details
            f.write("="*80 + "\n")
            f.write("RELEVANT ARTICLES\n")
            f.write("="*80 + "\n\n")
            
            # Sort articles by relevance score
            sorted_articles = sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            for i, article in enumerate(sorted_articles, 1):
                f.write(f"Article {i}:\n")
                f.write(f"Title: {article['title']}\n")
                f.write(f"Source: {article['source']}\n")
                f.write(f"Published: {article['published_date'].strftime('%Y-%m-%d')}\n")
                f.write(f"Topic: {article['topic_category']}\n")
                f.write(f"Relevance Score: {article.get('relevance_score', 0):.3f}\n")
                f.write(f"Sentiment Score: {article.get('sentiment_score', 0):.3f}\n")
                f.write(f"URL: {article['url']}\n")
                f.write("-"*80 + "\n\n")
        
        print(f"\nVisualizations and analysis have been saved in directory: {runtime_dir}")

    def final_segment_titles(self, all_titles: list) -> dict:
        """Assign each title to its best-matching category using detect_topic_category (best match only)."""
        final_categorized = defaultdict(list)
        for title, _ in all_titles:
            best_category = self.detect_topic_category(title)
            final_categorized[best_category].append(title)
        return final_categorized

# Example usage and demonstration
def main():
    """Demonstrate the Technology News IR System with Google Scholar"""
    print("Initializing Technology News IR System...")
    print("Using web scraping from Google Scholar")
    print()
    
    ir_system = TechNewsIRSystem()
    
    # Dictionary to store all category titles
    all_category_titles = defaultdict(list)
    
    # Example queries based on current tech topics
    queries = list(ir_system.tech_topics.keys(),)
    print(f"\n{'='*80}")
    print("PERFORMING SEARCH QUERIES FOR MAIN CATEGORIES")
    print('='*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Searching for: '{query}'")
        print('-' * 60)
        
        # Perform search
        results = ir_system.search_and_analyze(query, num_articles=15)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            continue
        
        # Display results summary
        print(f"Query: {results['query']}")
        print(f"Expanded Query: {results['expanded_query']}")
        print(f"Processing Time: {results['processing_time']:.2f} seconds")
        print(f"Number of Results: {results['num_results']}")
        
        # Accumulate category titles
        for category, titles in results['category_titles'].items():
            all_category_titles[category].extend(titles)
        
        # Add delay between queries to be respectful to servers
        if i < len(queries):
            time.sleep(5)
    
    # FINAL SEGMENTATION: Assign each title to its best-matching category and write to .txt files
    print(f"\n{'='*80}")
    print("FINAL SEGMENTATION OF TITLES INTO BEST-MATCH CATEGORIES")
    print('='*80)
    all_titles = []
    for category, titles in all_category_titles.items():
        for title in titles:
            all_titles.append((title, category))
    final_categorized_titles = ir_system.final_segment_titles(all_titles)
    output_dir = "final_categorized_titles"
    os.makedirs(output_dir, exist_ok=True)
    for category, titles in final_categorized_titles.items():
        filename = os.path.join(output_dir, f"{category.lower().replace('/', '_')}_titles.txt")
        with open(filename, "w", encoding="utf-8") as f:
            for title in titles:
                f.write(title + "\n")
    print(f"Final categorized titles have been saved in: {output_dir}")
    
    # Create consolidated analysis table
    print(f"\n{'='*80}")
    print("CONSOLIDATED CATEGORY ANALYSIS")
    print('='*80)
    
    analysis_table, titles_dir = ir_system.create_category_analysis_table(all_category_titles)
    print("\nCategory Analysis Table:")
    print(analysis_table.to_string(index=False))
    
    # Create visualizations
    print("\nCreating category visualizations...")
    ir_system.create_category_visualizations(analysis_table, titles_dir)
    
    # Save analysis table to CSV
    csv_filename = os.path.join(titles_dir, "category_analysis.csv")
    analysis_table.to_csv(csv_filename, index=False)
    print(f"\nAnalysis table has been saved to: {csv_filename}")
    
    # Create additional visualizations from CSV data
    print("\nCreating additional visualizations from CSV data...")
    ir_system.create_additional_visualizations(csv_filename, titles_dir)
    
    print(f"Category titles and visualizations have been saved in: {titles_dir}")
    
    print(f"\n{'='*80}")
    print("SYSTEM DEMONSTRATION COMPLETE")
    print('='*80)

if __name__ == "__main__":
    main()

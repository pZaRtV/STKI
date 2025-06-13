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
import sqlite3
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
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
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
                     'ai', 'ml', 'algorithm', 'automation', 'chatgpt', 'openai'],
            'Blockchain': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'crypto', 'defi', 
                          'nft', 'web3', 'smart contract'],
            'Cloud Computing': ['cloud', 'aws', 'azure', 'google cloud', 'saas', 'paas', 'iaas', 
                               'serverless', 'kubernetes'],
            'Cybersecurity': ['cybersecurity', 'security', 'hack', 'breach', 'malware', 'ransomware', 
                             'phishing', 'encryption'],
            'Mobile Tech': ['mobile', 'smartphone', 'ios', 'android', 'app', 'mobile app', 'tablet'],
            'IoT': ['iot', 'internet of things', 'smart home', 'connected device', 'sensor'],
            'Data Science': ['data science', 'big data', 'analytics', 'data mining', 'visualization'],
            'Programming': ['python', 'javascript', 'java', 'programming', 'software development', 
                           'coding', 'framework']
        }
        
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for storing search history and results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                num_results INTEGER,
                processing_time REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                title TEXT,
                content TEXT,
                url TEXT,
                published_date DATETIME,
                source TEXT,
                topic_category TEXT,
                relevance_score REAL,
                sentiment_score REAL,
                FOREIGN KEY (search_id) REFERENCES search_history (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_expansions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                original_query TEXT,
                expanded_query TEXT,
                expansion_terms TEXT,
                FOREIGN KEY (search_id) REFERENCES search_history (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
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
        Fetch real-time technology news from MIT Technology Review and Ars Technica
        """
        all_articles = []
        
        # RSS feeds for tech news sources
        rss_feeds = {
            'MIT Technology Review': 'https://www.technologyreview.com/feed/',
            'Ars Technica': 'https://feeds.arstechnica.com/arstechnica/index'
        }
        
        print("Fetching articles from RSS feeds...")
        
        # Fetch from RSS feeds
        for source, rss_url in rss_feeds.items():
            print(f"Fetching from {source}...")
            articles = self.fetch_rss_feed(rss_url, source)
            
            # Enhance articles with full content for better matching
            for article in articles[:10]:  # Limit to avoid too many requests
                if article['url'] and len(article['content']) < 500:  # Only scrape if content is short
                    full_content = self.scrape_article_content(article['url'])
                    if full_content:
                        article['content'] = full_content
                    
                    # Add small delay to be respectful
                    time.sleep(0.5)
            
            all_articles.extend(articles)
        
        # Filter articles based on query relevance if query is provided
        if query and query.strip():
            relevant_articles = []
            query_terms = query.lower().split()
            
            for article in all_articles:
                title_content = (article['title'] + ' ' + article['content']).lower()
                # Check if any query term appears in title or content
                relevance_score = sum(1 for term in query_terms if term in title_content)
                
                if relevance_score > 0:
                    article['initial_relevance'] = relevance_score
                    relevant_articles.append(article)
            
            # Sort by initial relevance and published date
            relevant_articles.sort(key=lambda x: (x.get('initial_relevance', 0), x['published_date']), reverse=True)
            return relevant_articles[:num_articles]
        
        # If no query, return recent articles
        all_articles.sort(key=lambda x: x['published_date'], reverse=True)
        return all_articles[:num_articles]
    
    def detect_topic_category(self, text: str) -> str:
        """Detect the primary technology topic category"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.tech_topics.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
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
    
    def search_and_analyze(self, query: str, num_articles: int = 20) -> Dict:
        """Main search function with comprehensive analysis"""
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
        
        # Store in database
        search_id = self.store_search_results(query, processed_articles, processing_time, 
                                            expanded_query, expansion_terms)
        
        # Generate analysis
        analysis = self.generate_analysis(processed_articles)
        
        return {
            'search_id': search_id,
            'query': query,
            'expanded_query': expanded_query,
            'expansion_terms': expansion_terms,
            'num_results': len(processed_articles),
            'processing_time': processing_time,
            'articles': processed_articles,
            'analysis': analysis
        }
    
    def store_search_results(self, query: str, articles: List[Dict], processing_time: float,
                           expanded_query: str, expansion_terms: List[str]) -> int:
        """Store search results in local database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert search history
        cursor.execute('''
            INSERT INTO search_history (query, num_results, processing_time)
            VALUES (?, ?, ?)
        ''', (query, len(articles), processing_time))
        
        search_id = cursor.lastrowid
        
        # Insert articles
        for article in articles:
            cursor.execute('''
                INSERT INTO news_articles 
                (search_id, title, content, url, published_date, source, topic_category, 
                 relevance_score, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                search_id, article['title'], article['content'], article['url'],
                article['published_date'], article['source'], article['topic_category'],
                article.get('relevance_score', 0.0), article.get('sentiment_score', 0.0)
            ))
        
        # Insert query expansion
        cursor.execute('''
            INSERT INTO query_expansions (search_id, original_query, expanded_query, expansion_terms)
            VALUES (?, ?, ?, ?)
        ''', (search_id, query, expanded_query, ','.join(expansion_terms)))
        
        conn.commit()
        conn.close()
        
        return search_id
    
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
    
    def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent search history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, query, timestamp, num_results, processing_time
            FROM search_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'search_id': row[0],
                'query': row[1],
                'timestamp': row[2],
                'num_results': row[3],
                'processing_time': row[4]
            })
        
        conn.close()
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
        
        # 2. Sentiment Distribution Plot
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
        
        # 3. Relevance Scores Plot
        plt.figure(figsize=(12, 8))
        relevance_scores = [article.get('relevance_score', 0.0) for article in articles]
        article_titles = [article['title'][:30] + '...' if len(article['title']) > 30 
                         else article['title'] for article in articles[:10]]
        plt.barh(range(len(article_titles)), relevance_scores[:10], color='orange')
        plt.yticks(range(len(article_titles)), article_titles, fontsize=8)
        plt.title(f'Top 10 Articles by Relevance for Query: "{query}"')
        plt.xlabel('Relevance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(runtime_dir, f"{base_filename}_relevance_scores.png"))
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
        
        # 5. Word Cloud Plot
        if articles:
            top_articles_text = ' '.join([article['title'] + ' ' + article['content'] 
                                        for article in articles[:5]])
            processed_text = self.preprocess_text(top_articles_text)
            
            if processed_text:
                plt.figure(figsize=(12, 6))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud of Top Articles for Query: "{query}"')
                plt.tight_layout()
                plt.savefig(os.path.join(runtime_dir, f"{base_filename}_wordcloud.png"))
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
                f.write(f"Published: {article['published_date'].strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"Topic: {article['topic_category']}\n")
                f.write(f"Relevance Score: {article.get('relevance_score', 0):.3f}\n")
                f.write(f"Sentiment Score: {article.get('sentiment_score', 0):.3f}\n")
                f.write(f"URL: {article['url']}\n")
                f.write("-"*80 + "\n\n")
        
        print(f"\nVisualizations and analysis have been saved in directory: {runtime_dir}")

# Example usage and demonstration
def main():
    """Demonstrate the Real-Time Technology News IR System with RSS/Scraping"""
    print("Initializing Real-Time Technology News IR System...")
    print("Using RSS feeds and web scraping from:")
    print("   • MIT Technology Review")
    print("   • Ars Technica")
    print()
    
    ir_system = TechNewsIRSystem()
    
    # Get trending topics first
    print(" Analyzing current trending topics...")
    trending = ir_system.get_trending_topics()
    if trending:
        print("\n Current Trending Topics:")
        for i, topic in enumerate(trending[:10], 1):
            print(f"   {i}. {topic['term']} ({topic['category']}) - Score: {topic['score']:.3f}")
    
    # Example queries based on current tech topics
    queries = [
        "artificial intelligence",
        "cybersecurity",
        "quantum computing",
        "machine learning",
        "blockchain"
    ]
    
    print(f"\n{'='*80}")
    print(" PERFORMING SEARCH QUERIES")
    print('='*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Searching for: '{query}'")
        print('-' * 60)
        
        # Perform search
        results = ir_system.search_and_analyze(query, num_articles=15)
        
        if 'error' in results:
            print(f" Error: {results['error']}")
            continue
        
        # Display results summary
        print(f" Query: {results['query']}")
        print(f" Expanded Query: {results['expanded_query']}")
        print(f"  Processing Time: {results['processing_time']:.2f} seconds")
        print(f" Number of Results: {results['num_results']}")
        
        if results['num_results'] > 0:
            print(f"\n Top 3 Most Relevant Articles:")
            for j, article in enumerate(results['articles'][:3], 1):
                print(f"\n   {j}.  {article['title'][:80]}{'...' if len(article['title']) > 80 else ''}")
                print(f"       Source: {article['source']}")
                print(f"       Topic: {article['topic_category']}")
                print(f"       Relevance: {article.get('relevance_score', 0):.3f}")
                print(f"       Sentiment: {article.get('sentiment_score', 0):.3f}")
                print(f"       Published: {article['published_date'].strftime('%Y-%m-%d %H:%M')}")
                print(f"       URL: {article['url'][:60]}{'...' if len(article['url']) > 60 else ''}")
            
            print(f"\n Analysis Summary:")
            analysis = results['analysis']
            print(f"    Average Relevance: {analysis.get('average_relevance', 0):.3f}")
            print(f"    Average Sentiment: {analysis.get('average_sentiment', 0):.3f}")
            print(f"    Recent Articles (24h): {analysis.get('recent_articles_count', 0)}")
            print(f"     Topics Found: {list(analysis.get('topic_distribution', {}).keys())}")
            
            # Create visualizations for the first few queries
            if i <= 2:  # Only visualize first 2 queries to avoid too many plots
                print(f"   Generating visualizations...")
                ir_system.visualize_results(results)
        else:
            print("   No relevant articles found for this query.")
        
        # Add delay between queries to be respectful to servers
        if i < len(queries):
            time.sleep(2)
    
    # Show search history
    print(f"\n{'='*80}")
    print(" RECENT SEARCH HISTORY")
    print('='*80)
    history = ir_system.get_search_history(limit=10)
    if history:
        for i, search in enumerate(history, 1):
            print(f"{i:2d}. 🔍 '{search['query']}' | "
                  f" {search['num_results']} results | "
                  f" {search['processing_time']:.2f}s | "
                  f" {search['timestamp']}")
    else:
        print("No search history found.")
    
    print(f"\n{'='*80}")
    print("SYSTEM DEMONSTRATION COMPLETE")
    print(f"All search data has been stored in: {ir_system.db_path}")
    print("You can now search for any technology topic!")
    print('='*80)

if __name__ == "__main__":
    main()

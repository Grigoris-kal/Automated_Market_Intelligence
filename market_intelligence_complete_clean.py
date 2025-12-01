# Cell 1: Installation & Imports - UPDATED WITH SCRAPY + SELENIUM
# Install required packages (run once)
# !pip install requests beautifulsoup4 pandas openpyxl schedule transformers torch
# !pip install textblob vaderSentiment spacy
# !pip install matplotlib seaborn
# !pip install scrapy selenium webdriver-manager
# !python -m spacy download en_core_web_sm

import json
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import schedule
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import warnings
warnings.filterwarnings('ignore')

# New imports for enhanced intelligence and visualization
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from matplotlib.patches import Patch
import io
from PIL import Image

# NEW: Scrapy and Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import twisted.internet
twisted.internet.reactor = None  # Prevent reactor restart issue

print("✅ All packages imported successfully! (Including Scrapy + Selenium)")


# Cell 2: Configuration Loaders - UPDATED
# Load configuration files
def load_companies_config():
    try:
        with open('config/companies_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ companies_config.json not found. Please create it first.")
        return None

def load_scraping_templates():
    try:
        with open('config/scraping_templates.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ scraping_templates.json not found. Please create it first.")
        return None

def load_email_config():
    try:
        with open('config/email_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ email_config.json not found. Please create from template.")
        return None

def load_analysis_period_config():
    """Load analysis period configuration"""
    try:
        with open('config/analysis_period_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ analysis_period_config.json not found. Using defaults.")
        return {'scraping_frequency_days': 2}

def load_social_media_config():
    """Load social media handles configuration"""
    try:
        with open('config/social_media_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ social_media_config.json not found. Social media features disabled.")
        return None

def load_publications_config():
    """Load professional publications configuration"""
    try:
        with open('config/publications_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ publications_config.json not found. Professional publications features disabled.")
        return None

# Load configurations
companies_config = load_companies_config()
scraping_templates = load_scraping_templates()
email_config = load_email_config()
analysis_period_config = load_analysis_period_config()
social_media_config = load_social_media_config()
publications_config = load_publications_config()

if companies_config and scraping_templates:
    print(f"✅ Loaded {len(companies_config['companies'])} companies")
    print(f"✅ Loaded {len(scraping_templates)} template categories")
    print(f"✅ Loaded analysis period config: {analysis_period_config}")
    if social_media_config:
        print(f"✅ Loaded social media config for {len(social_media_config.get('platforms', []))} platforms")
    if publications_config:
        print(f"✅ Loaded publications config for {len(publications_config.get('publications', []))} publications")


# Cell 3: Enhanced Business Intelligence Patterns - UPDATED
# Comprehensive patterns for sales and business activity detection
SALE_KEYWORDS = [
    # English
    'sale', 'offer', 'promo', 'discount', 'clearance', 'deal', 'bargain', 
    'save', 'special', 'limited', 'reduced', 'markdown', 'closeout', 'price drop',
    # French
    'rabais', 'soldes', 'promotion', 
    # German
    'angebot', 'aktion', 'reduziert',
    # Spanish
    'descuento', 'oferta', 'rebaja'
]

BUSINESS_ACTIVITY_PATTERNS = {
    'events': ['exhibition', 'trade show', 'event', 'conference', 'expo', 'fair', 'summit'],
    'news': ['news', 'announcement', 'update', 'press release', 'media'], 
    'launches': ['new', 'launch', 'introducing', 'now available', 'unveil', 'release'],
    'partnerships': ['partnership', 'collaboration', 'joint venture', 'alliance', 'cooperation'],
    'awards': ['award', 'recognition', 'achievement', 'prize', 'honor'],
    'expansion': ['expansion', 'growth', 'new location', 'opening', 'enter market']
}

SALE_PATTERNS = ['sale', 'offer', 'promo', 'discount', 'clearance', 'deal']
PRICE_PATTERNS = [r'\$\d+\.\d+\s*-\s*\$\d+\.\d+', r'\d+%?\s*off', r'save\s*\$\d+', r'\d+%?\s*discount', r'reduced\s*from\s*\$\d+']

# NEW: Professional Publications and Social Media Patterns
PROFESSIONAL_PUBLICATIONS = [
    'Bike Europe', 'Cycling Industry News', 'ECF', 'Bicycle Retailer',
    'Cycling Weekly', 'VeloNews', 'BikeBiz'
]

SOCIAL_MEDIA_PLATFORMS = {
    'twitter': {'domain': 'twitter.com', 'patterns': ['tweet', 'retweet', 'follow']},
    'instagram': {'domain': 'instagram.com', 'patterns': ['post', 'story', 'reel']},
    'linkedin': {'domain': 'linkedin.com', 'patterns': ['post', 'article', 'update']},
    'facebook': {'domain': 'facebook.com', 'patterns': ['post', 'share', 'like']}
}


# NEW CELL: Selenium Web Driver Manager
class SeleniumScraper:
    """Advanced scraper using Selenium for JavaScript-heavy sites and social media"""
    
    def __init__(self):
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        """Setup Chrome driver with optimal configuration"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            self.driver = webdriver.Chrome(
                ChromeDriverManager().install(),
                options=chrome_options
            )
            self.driver.set_page_load_timeout(30)
            print("✅ Selenium WebDriver initialized successfully")
            
        except Exception as e:
            print(f"❌ Selenium WebDriver initialization failed: {e}")
            self.driver = None
    
    def scrape_with_selenium(self, url, wait_for_element=None, timeout=10):
        """Scrape JavaScript-heavy websites using Selenium"""
        if not self.driver:
            print("❌ Selenium driver not available")
            return None
        
        try:
            print(f"🔍 Selenium scraping: {url}")
            self.driver.get(url)
            
            # Wait for specific element if provided
            if wait_for_element:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )
            else:
                # Default wait for body content
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            
            # Get page source after JavaScript execution
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            return {
                'url': url,
                'content': soup.get_text(strip=True, separator=' ')[:2000],
                'html': page_source,
                'timestamp': datetime.now().isoformat(),
                'scraping_method': 'selenium'
            }
            
        except TimeoutException:
            print(f"⏰ Timeout scraping {url}")
            return None
        except Exception as e:
            print(f"❌ Selenium scraping error for {url}: {e}")
            return None
    
    def scrape_social_media(self, platform, username):
        """Scrape social media profiles for activity and engagement"""
        if platform not in SOCIAL_MEDIA_PLATFORMS:
            print(f"❌ Unsupported platform: {platform}")
            return None
        
        platform_config = SOCIAL_MEDIA_PLATFORMS[platform]
        profile_url = f"https://{platform_config['domain']}/{username}"
        
        try:
            print(f"📱 Scraping {platform} profile: {username}")
            data = self.scrape_with_selenium(profile_url, wait_for_element="[data-testid='tweet']" if platform == 'twitter' else "article")
            
            if data:
                # Enhanced social media analysis
                data.update(self.analyze_social_media_content(data['html'], platform, username))
                data['platform'] = platform
                data['username'] = username
                
            return data
            
        except Exception as e:
            print(f"❌ Social media scraping error for {username} on {platform}: {e}")
            return None
    
    def analyze_social_media_content(self, html_content, platform, username):
        """Analyze social media content for engagement and activity patterns"""
        soup = BeautifulSoup(html_content, 'html.parser')
        analysis = {
            'post_count_estimate': 0,
            'engagement_indicators': [],
            'content_themes': [],
            'activity_level': 'low'
        }
        
        try:
            # Platform-specific analysis
            if platform == 'twitter':
                # Count tweets
                tweets = soup.find_all('article') or soup.find_all('[data-testid="tweet"]')
                analysis['post_count_estimate'] = len(tweets)
                
                # Engagement metrics
                likes = soup.find_all(text=re.compile(r'\d+\s+Like'))
                retweets = soup.find_all(text=re.compile(r'\d+\s+Retweet'))
                
            elif platform == 'instagram':
                # Post count from profile header
                post_count_elem = soup.find(text=re.compile(r'\d+\s+posts'))
                if post_count_elem:
                    analysis['post_count_estimate'] = int(re.search(r'(\d+)\s+posts', post_count_elem).group(1))
            
            # Determine activity level
            if analysis['post_count_estimate'] > 20:
                analysis['activity_level'] = 'high'
            elif analysis['post_count_estimate'] > 5:
                analysis['activity_level'] = 'medium'
                
        except Exception as e:
            print(f"⚠️ Social media analysis error: {e}")
        
        return analysis
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            print("✅ Selenium WebDriver closed")


# NEW CELL: Scrapy Spider for Professional Publications
class ProfessionalPublicationsSpider(scrapy.Spider):
    """Scrapy spider for scraping professional cycling publications"""
    
    name = "professional_publications"
    
    def __init__(self, publications=None, *args, **kwargs):
        super(ProfessionalPublicationsSpider, self).__init__(*args, **kwargs)
        self.publications = publications or PROFESSIONAL_PUBLICATIONS
        self.scraped_data = []
    
    def start_requests(self):
        """Generate requests for each publication"""
        publication_urls = {
            'Bike Europe': 'https://www.bike-eu.com',
            'Cycling Industry News': 'https://cyclingindustry.news',
            'ECF': 'https://ecf.com/news',
            'Bicycle Retailer': 'https://www.bicycleretailer.com',
            'Cycling Weekly': 'https://www.cyclingweekly.com/news',
            'VeloNews': 'https://www.velonews.com',
            'BikeBiz': 'https://www.bikebiz.com'
        }
        
        for publication, url in publication_urls.items():
            if publication in self.publications:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_publication,
                    meta={'publication': publication}
                )
    
    def parse_publication(self, response):
        """Parse publication homepage for company mentions"""
        publication = response.meta['publication']
        
        # Look for company mentions in article titles and summaries
        articles = response.css('article, .news-item, .post, .story')
        
        for article in articles[:10]:  # Limit to first 10 articles
            title = article.css('h1, h2, h3::text').get()
            summary = article.css('p::text').get()
            link = article.css('a::attr(href)').get()
            
            if title or summary:
                content = f"{title or ''} {summary or ''}"
                
                # Check for company mentions
                mentioned_companies = self.detect_company_mentions(content)
                
                if mentioned_companies:
                    article_data = {
                        'publication': publication,
                        'title': title,
                        'summary': summary[:200] if summary else '',
                        'url': response.urljoin(link) if link else response.url,
                        'mentioned_companies': mentioned_companies,
                        'scraped_at': datetime.now().isoformat(),
                        'content_type': 'professional_publication'
                    }
                    
                    self.scraped_data.append(article_data)
        
    def detect_company_mentions(self, content):
        """Detect company mentions in publication content"""
        if not companies_config:
            return []
        
        mentioned_companies = []
        content_lower = content.lower()
        
        for company_id, company_info in companies_config['companies'].items():
            company_name = company_info['official_name'].lower()
            # Check for company name in content
            if company_name in content_lower:
                mentioned_companies.append(company_info['official_name'])
            # Also check for common abbreviations or brand names
            elif 'brand_name' in company_info and company_info['brand_name'].lower() in content_lower:
                mentioned_companies.append(company_info['official_name'])
        
        return mentioned_companies


class ScrapyManager:
    """Manager for Scrapy-based professional publication scraping"""
    
    def __init__(self):
        self.process = None
        self.spider = None
    
    def scrape_publications(self, publications=None):
        """Scrape professional publications using Scrapy"""
        try:
            print("📰 Starting professional publications scraping with Scrapy...")
            
            # Configure Scrapy settings
            settings = get_project_settings()
            settings.update({
                'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'CONCURRENT_REQUESTS': 2,
                'DOWNLOAD_DELAY': 2,
                'AUTOTHROTTLE_ENABLED': True,
                'LOG_LEVEL': 'ERROR'
            })
            
            self.process = CrawlerProcess(settings)
            self.spider = ProfessionalPublicationsSpider(publications)
            
            # Run the spider
            deferred = self.process.crawl(self.spider)
            deferred.addCallback(self.on_scraping_finished)
            
            # Start the reactor
            self.process.start(stop_after_crawl=True)
            
            return self.spider.scraped_data
            
        except Exception as e:
            print(f"❌ Scrapy scraping error: {e}")
            return []
    
    def on_scraping_finished(self, result):
        """Callback when Scrapy scraping finishes"""
        print(f"✅ Professional publications scraping completed: {len(self.spider.scraped_data)} articles found")
        return self.spider.scraped_data


# Cell 4: Enhanced Intelligence Detection Functions - UPDATED WITH SOCIAL MEDIA
def comprehensive_sale_detection(html_content, company_name):
    """
    Advanced sale detection using multiple pattern matching strategies
    """
    if not html_content or len(html_content.strip()) < 100:
        return {'sale_detected': False, 'confidence': 0, 'details': 'Insufficient content'}
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements for cleaner text analysis
    for element in soup(["script", "style"]):
        element.decompose()
    
    # Phase 1: Attribute-based detection (CSS wildcards)
    attribute_matches = []
    for pattern in SALE_PATTERNS:
        attribute_matches.extend(soup.select(f'[class*="{pattern}"]'))
        attribute_matches.extend(soup.select(f'[id*="{pattern}"]'))
        attribute_matches.extend(soup.select(f'[data-*="{pattern}"]'))
    
    # Phase 2: Text content detection WITH WILDCARD ENHANCEMENT
    text_matches = []
    page_text = soup.get_text().lower()
    
    # NEW: Wildcard patterns for common sales terms
    wildcard_patterns = [
        r'\w*sale\w*',      # Matches "summersale", "saleevent", "sales"
        r'\w*promo\w*',     # Matches "promocode", "winterpromo", "promotion"  
        r'\w*offer\w*',     # Matches "specialoffer", "offerdetails", "offering"
        r'\w*discount\w*',  # Matches "discountcode", "megadiscount"
        r'\w*clearance\w*', # Matches "clearanceevent", "clearanceitems"
        r'\w*liquidation\w*' # Matches "liquidationSale", "liquidationEvent"
    ]
    
    # Combine original keyword matching WITH wildcard matching
    for keyword in SALE_KEYWORDS:
        # Original exact matching (preserved)
        if keyword in page_text:
            elements_with_keyword = soup.find_all(text=re.compile(re.escape(keyword), re.IGNORECASE))
            text_matches.extend(elements_with_keyword)
    
    # NEW: Add wildcard pattern matching
    for pattern in wildcard_patterns:
        wildcard_elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        text_matches.extend(wildcard_elements)
    
    # Phase 3: Price pattern detection
    price_matches = []
    for price_pattern in PRICE_PATTERNS:
        price_elements = soup.find_all(text=re.compile(price_pattern, re.IGNORECASE))
        price_matches.extend(price_elements)
    
    # ⭐⭐⭐ IMPROVED CONFIDENCE SCORING SYSTEM ⭐⭐⭐
    def calculate_quality_score(matches, strong_threshold=3):
        """Calculate quality score based on evidence strength"""
        if len(matches) == 0:
            return 0
        elif len(matches) == 1:
            return 1  # Weak evidence
        elif len(matches) == 2:
            return 2  # Moderate evidence
        elif len(matches) >= strong_threshold:
            return 3  # Strong evidence
        else:
            return min(len(matches), 3)
    
    # Quality-based scoring
    confidence_factors = {
        'attribute_quality': calculate_quality_score(attribute_matches),
        'text_quality': calculate_quality_score(text_matches, strong_threshold=4),  # Text needs more evidence
        'price_quality': calculate_quality_score(price_matches),
        'multiple_methods': 2 if (len(attribute_matches) >= 2 and len(text_matches) >= 2) else 0,  # Stricter
        'strong_text_evidence': 2 if len(text_matches) >= 4 else 0,  # Bonus for strong text evidence
        'combined_evidence': 2 if (len(attribute_matches) >= 1 and len(text_matches) >= 1 and len(price_matches) >= 1) else 0  # All three methods
    }
    
    total_score = sum(confidence_factors.values())
    max_possible = 14  # 3 + 3 + 3 + 2 + 2 + 2
    confidence = min(int((total_score / max_possible) * 100), 100)
    
    # ⭐⭐⭐ BETTER THRESHOLD ⭐⭐⭐
    sale_detected = confidence >= 60  # Now requires stronger evidence
    
    # Get unique sale keywords found
    unique_keywords = set()
    for match in text_matches[:5]:  # Limit to first 5 matches
        text_lower = match.get_text().lower()
        for keyword in SALE_KEYWORDS:
            if keyword in text_lower:
                unique_keywords.add(keyword)
    
    return {
        'sale_detected': sale_detected,
        'confidence': confidence,
        'sale_keywords_found': list(unique_keywords)[:5],  # Top 5 keywords
        'detection_breakdown': {
            'attribute_elements': len(attribute_matches),
            'text_mentions': len(text_matches),
            'price_indicators': len(price_matches),
            'confidence_factors': confidence_factors,
            'total_score': total_score,
            'max_possible': max_possible
        },
        'sample_elements': [match.get_text().strip() for match in (attribute_matches + text_matches)[:3]]  # Sample evidence
    }


# =============================================================================
# STANDARDIZED MARKET REALITY DASHBOARD FUNCTIONS
# =============================================================================

def analyze_strategic_postures(companies_data):
    """
    Categorize each company's current market position
    Returns standardized posture assessment for each company
    """
    postures = {}
    
    for company_name, company_data in companies_data.items():
        activities = company_data.get('business_activities', {})
        sales_intel = company_data.get('sales_intelligence', {})
        
        # Calculate activity scores
        sales_aggression = sales_intel.get('confidence', 0) / 100
        launch_activity = len(activities.get('launches', []))
        partnership_activity = len(activities.get('partnerships', []))
        event_activity = len(activities.get('events', []))
        total_activities = sum(len(acts) for acts in activities.values())
        
        # Determine strategic posture
        if sales_aggression > 0.7 and launch_activity >= 2:
            posture = "Aggressive"
        elif partnership_activity >= 2 and event_activity >= 2:
            posture = "Collaborative"
        elif sales_aggression > 0.6 and total_activities <= 2:
            posture = "Defensive"
        elif launch_activity >= 3 or partnership_activity >= 3:
            posture = "Focused"
        elif total_activities == 0:
            posture = "Dormant"
        else:
            posture = "Balanced"
            
        postures[company_name] = {
            'posture': posture,
            'sales_aggression': sales_aggression,
            'launch_intensity': launch_activity,
            'partnership_density': partnership_activity,
            'total_activities': total_activities
        }
    
    return postures

def detect_live_market_movements(companies_data):
    """
    Identify active strategic moves in the market
    Returns standardized movement detection
    """
    movements = {
        'sales_blitzes': [],
        'product_launches': [],
        'partnership_announcements': [],
        'expansion_moves': [],
        'high_impact_events': []
    }
    
    for company_name, company_data in companies_data.items():
        activities = company_data.get('business_activities', {})
        sales_intel = company_data.get('sales_intelligence', {})
        
        # Detect sales blitzes (high confidence sales)
        if sales_intel.get('confidence', 0) >= 70:
            movements['sales_blitzes'].append({
                'company': company_name,
                'confidence': sales_intel.get('confidence', 0),
                'keywords': sales_intel.get('sale_keywords_found', [])
            })
        
        # Detect product launches
        if len(activities.get('launches', [])) >= 1:
            movements['product_launches'].append({
                'company': company_name,
                'launch_count': len(activities.get('launches', [])),
                'recent_launches': activities.get('launches', [])[:2]
            })
        
        # Detect partnerships
        if len(activities.get('partnerships', [])) >= 1:
            movements['partnership_announcements'].append({
                'company': company_name,
                'partnership_count': len(activities.get('partnerships', [])),
                'recent_partnerships': activities.get('partnerships', [])[:2]
            })
        
        # Detect expansion moves
        if len(activities.get('expansion', [])) >= 1:
            movements['expansion_moves'].append({
                'company': company_name,
                'expansion_count': len(activities.get('expansion', [])),
                'expansion_details': activities.get('expansion', [])[:2]
            })
        
        # Detect high-impact events
        if len(activities.get('events', [])) >= 2:
            movements['high_impact_events'].append({
                'company': company_name,
                'event_count': len(activities.get('events', [])),
                'major_events': activities.get('events', [])[:2]
            })
    
    return movements

def assess_competitive_threats(companies_data, your_company=None):
    """
    Evaluate immediate competitive threats with standardized scoring
    """
    threats = {
        'direct_attacks': [],
        'adjacent_expansions': [], 
        'market_noise': [],
        'threat_levels': {}
    }
    
    for company_name, company_data in companies_data.items():
        # Skip your own company if specified
        if your_company and company_name == your_company:
            continue
            
        activities = company_data.get('business_activities', {})
        sales_intel = company_data.get('sales_intelligence', {})
        
        threat_score = 0
        threat_reasons = []
        
        # Direct attacks: High sales aggression + same market focus
        if sales_intel.get('confidence', 0) >= 60:
            threat_score += 3
            threat_reasons.append("Aggressive sales campaign")
        
        # Product launches in core segments
        if len(activities.get('launches', [])) >= 2:
            threat_score += 2
            threat_reasons.append("Multiple product launches")
        
        # Geographic expansion
        if len(activities.get('expansion', [])) >= 1:
            threat_score += 2
            threat_reasons.append("Market expansion")
        
        # High overall activity
        total_activities = sum(len(acts) for acts in activities.values())
        if total_activities >= 5:
            threat_score += 1
            threat_reasons.append("High market activity")
        
        # Categorize threat level
        if threat_score >= 5:
            threat_level = "High"
            threats['direct_attacks'].append(company_name)
        elif threat_score >= 3:
            threat_level = "Medium" 
            threats['adjacent_expansions'].append(company_name)
        elif threat_score >= 1:
            threat_level = "Low"
            threats['market_noise'].append(company_name)
        else:
            threat_level = "None"
        
        threats['threat_levels'][company_name] = {
            'level': threat_level,
            'score': threat_score,
            'reasons': threat_reasons,
            'total_activities': total_activities
        }
    
    return threats

def standardized_sentiment_analysis(companies_data):
    """
    Consistent sentiment scoring across all companies
    """
    sentiment_results = {}
    
    for company_name, company_data in companies_data.items():
        sentiment_data = company_data.get('sentiment', {})
        
        # Standardize sentiment scoring (0-100 scale)
        raw_sentiment = sentiment_data.get('overall_sentiment_score', 0)
        
        # Convert to standardized 0-100 scale
        if isinstance(raw_sentiment, float) and raw_sentiment <= 1.0:
            standardized_score = int(raw_sentiment * 100)
        else:
            standardized_score = min(max(int(raw_sentiment), 0), 100)
        
        # Categorize sentiment
        if standardized_score >= 70:
            sentiment_category = "Positive"
        elif standardized_score >= 40:
            sentiment_category = "Neutral" 
        else:
            sentiment_category = "Negative"
        
        sentiment_results[company_name] = {
            'score': standardized_score,
            'category': sentiment_category,
            'positive_mentions': sentiment_data.get('positive_mentions', 0),
            'negative_mentions': sentiment_data.get('negative_mentions', 0),
            'total_mentions': sentiment_data.get('total_mentions', 0)
        }
    
    return sentiment_results

def standardized_share_of_voice(companies_data):
    """
    Consistent market presence measurement across all companies
    """
    # Calculate total mentions across all companies
    total_mentions = 0
    company_mentions = {}
    
    for company_name, company_data in companies_data.items():
        sentiment_data = company_data.get('sentiment', {})
        activities = company_data.get('business_activities', {})
        
        # Count total activity mentions
        activity_mentions = sum(len(acts) for acts in activities.values())
        sentiment_mentions = sentiment_data.get('total_mentions', 0)
        
        total_company_mentions = activity_mentions + sentiment_mentions
        company_mentions[company_name] = total_company_mentions
        total_mentions += total_company_mentions
    
    # Calculate share of voice percentages
    share_of_voice = {}
    for company_name, mentions in company_mentions.items():
        if total_mentions > 0:
            percentage = (mentions / total_mentions) * 100
        else:
            percentage = 0
        
        # Categorize market presence
        if percentage >= 30:
            presence = "Dominant"
        elif percentage >= 15:
            presence = "Strong"
        elif percentage >= 5:
            presence = "Moderate"
        else:
            presence = "Limited"
        
        share_of_voice[company_name] = {
            'percentage': round(percentage, 2),
            'presence_level': presence,
            'total_mentions': mentions
        }
    
    return share_of_voice

def generate_market_reality_report(companies_data, your_company=None):
    """
    Generate comprehensive standardized market reality report
    """
    # Run all standardized analyses
    postures = analyze_strategic_postures(companies_data)
    movements = detect_live_market_movements(companies_data)
    threats = assess_competitive_threats(companies_data, your_company)
    sentiment = standardized_sentiment_analysis(companies_data)
    sov = standardized_share_of_voice(companies_data)
    
    # Compile comprehensive report
    market_reality_report = {
        'timestamp': datetime.now().isoformat(),
        'companies_analyzed': list(companies_data.keys()),
        'strategic_postures': postures,
        'market_movements': movements,
        'competitive_threats': threats,
        'sentiment_analysis': sentiment,
        'share_of_voice': sov,
        'summary_metrics': {
            'total_companies': len(companies_data),
            'companies_with_sales': len([c for c, data in companies_data.items() 
                                       if data.get('sales_intelligence', {}).get('sale_detected')]),
            'total_activities': sum(sum(len(acts) for acts in data.get('business_activities', {}).values()) 
                                  for data in companies_data.values()),
            'high_threat_companies': len(threats['direct_attacks'])
        }
    }
    
    return market_reality_report


def detect_business_activities(html_content, company_name):
    """
    Detect strategic business activities beyond just sales
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()
    
    activity_intel = {}
    
    # Detect various business activities
    for activity_type, keywords in BUSINESS_ACTIVITY_PATTERNS.items():
        detected_activities = []
        for keyword in keywords:
            elements = soup.find_all(text=re.compile(keyword, re.IGNORECASE))
            for element in elements[:2]:  # Limit to top 2 matches per keyword
                text = element.get_text().strip()
                if len(text) > 10 and len(text) < 500:  # Filter out tiny fragments and huge blocks
                    detected_activities.append({
                        'keyword': keyword,
                        'text_snippet': text[:100] + '...' if len(text) > 100 else text,
                        'confidence': min(len(text) / 5, 100)  # Simple confidence based on text length
                    })        
        activity_intel[activity_type] = detected_activities[:3]  # Top 3 per category
    
    return activity_intel

def generate_company_intelligence_summary(company_data):
    """Generate natural language business intelligence summary"""
    summary_parts = []
    company_name = company_data.get('company', 'Company')
    
    # Sales intelligence
    sales_intel = company_data.get('sales_intelligence', {})
    if sales_intel.get('sale_detected'):
        confidence = sales_intel.get('confidence', 0)
        keywords = sales_intel.get('sale_keywords_found', [])
        if confidence >= 70:
            summary_parts.append(f"running {', '.join(keywords[:2])} with {confidence}% confidence")
        else:
            summary_parts.append(f"possible {', '.join(keywords[:1])} activity ({confidence}% confidence)")
    
    # Business activities intelligence
    activities = company_data.get('business_activities', {})
    
    # Events & exhibitions
    if activities.get('events'):
        event_count = len(activities['events'])
        summary_parts.append(f"participating in {event_count} industry event{'s' if event_count > 1 else ''}")
    
    # Product launches
    if activities.get('launches'):
        launch_count = len(activities['launches'])
        summary_parts.append(f"launched {launch_count} new product{'s' if launch_count > 1 else ''}")
    
    # Partnerships
    if activities.get('partnerships'):
        partnership_count = len(activities['partnerships'])
        summary_parts.append(f"announced {partnership_count} partnership{'s' if partnership_count > 1 else ''}")
    
    # Expansion
    if activities.get('expansion'):
        expansion_count = len(activities['expansion'])
        summary_parts.append(f"expanding operations in {expansion_count} area{'s' if expansion_count > 1 else ''}")
    
    # Awards
    if activities.get('awards'):
        award_count = len(activities['awards'])
        summary_parts.append(f"received {award_count} award{'s' if award_count > 1 else ''}")
    
    # News
    if activities.get('news'):
        news_count = len(activities['news'])
        summary_parts.append(f"published {news_count} announcement{'s' if news_count > 1 else ''}")
    
    # Social media activity
    social_data = company_data.get('social_media_analysis', {})
    if social_data.get('activity_level') == 'high':
        summary_parts.append("high social media engagement")
    elif social_data.get('activity_level') == 'medium':
        summary_parts.append("moderate social media presence")
    
    # Professional publication mentions
    pub_mentions = company_data.get('publication_mentions', [])
    if pub_mentions:
        summary_parts.append(f"featured in {len(pub_mentions)} professional publication{'s' if len(pub_mentions) > 1 else ''}")
    
    # Sentiment context
    sentiment = company_data.get('sentiment', {}).get('overall_sentiment', 'neutral')
    if sentiment != 'neutral' and summary_parts:
        summary_parts.append(f"showing {sentiment} market sentiment")
    
    if summary_parts:
        return company_name + ": " + ". ".join(summary_parts) + "."
    else:
        return company_name + ": Limited strategic activity detected this period."


# Cell 5: Enhanced Data Collection Functions - UPDATED WITH SCRAPY + SELENIUM
class DataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        # NEW: Initialize Selenium and Scrapy
        self.selenium_scraper = SeleniumScraper()
        self.scrapy_manager = ScrapyManager()
        self.social_media_config = load_social_media_config()
        self.publications_config = load_publications_config()
    
    def scrape_website(self, url, company_name):
        """Generic website scraper with error handling"""
        try:
            print(f"🌐 Scraping: {company_name} - {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract relevant sections (customize per site structure)
            data = {
                'company': company_name,
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'title': self.extract_title(soup),
                'content': self.extract_content(soup),
                'metadata': self.extract_metadata(soup)
            }
            
            return data
            
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {str(e)}")
            return None

    def scrape_with_business_intelligence(self, url, company_name):
        """
        Enhanced scraping that includes sales and business activity intelligence
        """
        try:
            print(f"🎯 Enhanced scraping with business intelligence: {company_name}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract standard data
            data = {
                'company': company_name,
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'title': self.extract_title(soup),
                'content': self.extract_content(soup),
                'metadata': self.extract_metadata(soup)
            }
            
            # Add enhanced intelligence
            data['sales_intelligence'] = comprehensive_sale_detection(response.content, company_name)
            data['business_activities'] = detect_business_activities(response.content, company_name)
            data['intelligence_summary'] = generate_company_intelligence_summary(data)
            
            # NEW: Add social media analysis if configured
            if self.social_media_config:
                data['social_media_analysis'] = self.scrape_social_media_profiles(company_name)
            
            # NEW: Add professional publication mentions
            if self.publications_config:
                data['publication_mentions'] = self.get_publication_mentions(company_name)
            
            # Add evidence and sources
            data['evidence_sources'] = self.extract_evidence_sources(soup, data, url)
            
            # Enhance sentiment analysis with business context
            business_context = ""
            if data['sales_intelligence']['sale_detected']:
                business_context += f"SALE DETECTED: {', '.join(data['sales_intelligence']['sale_keywords_found'])}. "
            
            activity_types = [act for act, items in data['business_activities'].items() if items]
            if activity_types:
                business_context += f"ACTIVITIES: {', '.join(activity_types)}. "
            
            enhanced_content = business_context + data.get('content', '')
            
            # Analyze sentiment with enhanced context
            data['sentiment'] = sentiment_analyzer.analyze_sentiment(enhanced_content)
            data['key_phrases'] = sentiment_analyzer.extract_key_phrases(enhanced_content)
            
            return data
            
        except Exception as e:
            print(f"❌ Failed to scrape {url} with business intelligence: {str(e)}")
            return None

    def scrape_social_media_profiles(self, company_name):
        """Scrape social media profiles for the company"""
        if not self.social_media_config:
            return {}
        
        social_data = {}
        company_handles = self.social_media_config.get('company_handles', {}).get(company_name, {})
        
        for platform, username in company_handles.items():
            if platform in SOCIAL_MEDIA_PLATFORMS:
                platform_data = self.selenium_scraper.scrape_social_media(platform, username)
                if platform_data:
                    social_data[platform] = platform_data
        
        return social_data

    def get_publication_mentions(self, company_name):
        """Get mentions in professional publications"""
        if not self.publications_config:
            return []
        
        # This would be populated by the Scrapy spider
        # For now, return empty list - will be implemented in full scraping cycle
        return []

    def run_comprehensive_scraping(self, company_name, company_info):
        """Run comprehensive scraping including social media and publications"""
        comprehensive_data = []
        
        # 1. Main website scraping
        website_data = self.scrape_with_business_intelligence(
            company_info['website'], 
            company_name
        )
        if website_data:
            comprehensive_data.append(website_data)
        
        # 2. Social media scraping
        social_data = self.scrape_social_media_profiles(company_name)
        if social_data:
            comprehensive_data.append({
                'company': company_name,
                'content_type': 'social_media',
                'social_media_data': social_data,
                'timestamp': datetime.now().isoformat()
            })
        
        return comprehensive_data

    def run_publications_scraping(self):
        """Run professional publications scraping using Scrapy"""
        publications = self.publications_config.get('publications', []) if self.publications_config else []
        return self.scrapy_manager.scrape_publications(publications)

    def extract_evidence_sources(self, soup, data, url):
        """Extract specific evidence and sources for intelligence findings"""
        evidence = {
            'website_url': url,
            'sales_evidence': [],
            'activity_evidence': {},
            'social_media_sources': [],
            'publication_sources': []
        }
        
        # Sales evidence
        sales_intel = data.get('sales_intelligence', {})
        if sales_intel.get('sale_detected'):
            evidence['sales_evidence'] = sales_intel.get('sample_elements', [])[:2]
        
        # Business activity evidence
        activities = data.get('business_activities', {})
        for activity_type, items in activities.items():
            if items:
                evidence['activity_evidence'][activity_type] = [item['text_snippet'] for item in items[:2]]
        
        # Social media sources
        social_data = data.get('social_media_analysis', {})
        for platform, platform_data in social_data.items():
            if platform_data.get('url'):
                evidence['social_media_sources'].append({
                    'platform': platform,
                    'url': platform_data['url'],
                    'activity_level': platform_data.get('activity_level', 'unknown')
                })
        
        # Publication sources
        pub_mentions = data.get('publication_mentions', [])
        for mention in pub_mentions:
            evidence['publication_sources'].append({
                'publication': mention.get('publication'),
                'url': mention.get('url'),
                'title': mention.get('title', '')[:100]
            })
        
        return evidence
    
    def extract_title(self, soup):
        """Extract page title"""
        title = soup.find('title')
        return title.get_text().strip() if title else "No title"
    
    def extract_content(self, soup):
        """Extract main content - focus on news, blog, press sections"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', '.content', '.main-content', '#content',
            '.news-content', '.press-release', '.blog-post',
            'article', '.article-content'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content.get_text(strip=True, separator=' ')[:1000]  # Limit length
        
        # Fallback: get body text
        return soup.get_text(strip=True, separator=' ')[:1000]
    
    def extract_metadata(self, soup):
        """Extract metadata like publication date, author"""
        metadata = {}
        
        # Date extraction
        date_selectors = [
            'time', '.date', '.publish-date', '[property*="date"]'
        ]
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                metadata['date'] = date_elem.get_text().strip()
                break
        
        return metadata

    def close(self):
        """Clean up resources"""
        if self.selenium_scraper:
            self.selenium_scraper.close()

# Initialize collector
collector = DataCollector()
print("✅ Enhanced data collector with Scrapy + Selenium initialized")


# Cell 6: Sentiment Analysis Utilities (UNCHANGED)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """Comprehensive sentiment analysis using multiple methods"""
        if not text or len(text.strip()) < 10:
            return {'overall_sentiment': 'neutral', 'confidence': 0.0}
        
        # TextBlob sentiment
        blob = TextBlob(text)
        tb_polarity = blob.sentiment.polarity
        tb_subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Combined scoring
        combined_score = (tb_polarity + vader_scores['compound']) / 2
        
        # Determine sentiment category
        if combined_score >= 0.05:
            sentiment = 'positive'
            confidence = abs(combined_score)
        elif combined_score <= -0.05:
            sentiment = 'negative'
            confidence = abs(combined_score)
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(combined_score)
        
        return {
            'overall_sentiment': sentiment,
            'confidence': round(confidence, 3),
            'scores': {
                'textblob_polarity': round(tb_polarity, 3),
                'textblob_subjectivity': round(tb_subjectivity, 3),
                'vader_compound': round(vader_scores['compound'], 3),
                'vader_positive': round(vader_scores['pos'], 3),
                'vader_negative': round(vader_scores['neg'], 3),
                'combined_score': round(combined_score, 3)
            }
        }
    
    def extract_key_phrases(self, text, num_phrases=5):
        """Extract key phrases from text"""
        if not text:
            return []
        
        blob = TextBlob(text)
        nouns = blob.noun_phrases
        return list(dict.fromkeys(nouns))[:num_phrases]  # Remove duplicates

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()
print("✅ Sentiment analyzer initialized")


# Cell 7: Cache Management System (UNCHANGED)
class CacheManager:
    def __init__(self):
        self.cache_dir = 'data/cache'
        self.historical_dir = 'data/historical'
        
    def get_current_cache_file(self):
        """Get cache file path for current date"""
        current_date = datetime.now().strftime('%Y_%m_%d')
        return os.path.join(self.cache_dir, f'collected_{current_date}.json')
    
    def save_to_cache(self, data):
        """Save collected data to daily cache file"""
        cache_file = self.get_current_cache_file()
        
        # Load existing cache or create new
        existing_data = []
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        
        # Add new data
        existing_data.extend(data)
        
        # Save back to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(data)} items to cache: {cache_file}")
        return len(existing_data)
    
    def load_current_month_cache(self):
        """Load all cache files from current month"""
        current_month = datetime.now().strftime('%Y_%m')
        monthly_data = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(f'collected_{current_month}') and filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    monthly_data.extend(file_data)
        
        print(f"📂 Loaded {len(monthly_data)} total items from {current_month} cache")
        return monthly_data
    
    def archive_current_month(self):
        """Archive current month's cache to historical (BOTH JSON and Excel) and clear cache"""
        current_month = datetime.now().strftime('%Y_%m')
        monthly_data = self.load_current_month_cache()
        
        if monthly_data:
            # 1. Save to JSON historical (existing functionality)
            historical_file = os.path.join(self.historical_dir, f'{current_month}_complete.json')
            with open(historical_file, 'w', encoding='utf-8') as f:
                json.dump(monthly_data, f, indent=2, ensure_ascii=False)
            
            # 2. NEW: Also save COMPLETE raw data to Excel historical folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            excel_historical_dir = os.path.join(script_dir, 'Historical_Data_Market_Intelligence')
            os.makedirs(excel_historical_dir, exist_ok=True)
            
            # Create comprehensive Excel archive with ALL raw data
            formatted_month = datetime.now().strftime('%Y-%m')
            excel_filename = os.path.join(excel_historical_dir, f'Complete_Raw_Data_{formatted_month}.xlsx')
            
            # Convert all raw data to Excel with proper formatting
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # Create a flat structure for raw data
                flat_data = []
                for item in monthly_data:
                    flat_item = {
                        'company': item.get('company', ''),
                        'url': item.get('url', ''),
                        'timestamp': item.get('timestamp', ''),
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'sentiment_overall': item.get('sentiment', {}).get('overall_sentiment', '') if isinstance(item.get('sentiment', {}), dict) else '',
                        'sentiment_confidence': item.get('sentiment', {}).get('confidence', 0) if isinstance(item.get('sentiment', {}), dict) else 0,
                        'key_phrases': ', '.join(item.get('key_phrases', [])) if isinstance(item.get('key_phrases'), list) else '',
                        'metadata': str(item.get('metadata', {}))
                    }
                    flat_data.append(flat_item)
                
                df_raw = pd.DataFrame(flat_data)
                df_raw.to_excel(writer, sheet_name='Complete_Raw_Data', index=False)
                
                # Add summary statistics
                summary_data = {
                    'Metric': ['Total Records', 'Companies Covered', 'Date Range', 'Data Collection Period'],
                    'Value': [
                        len(monthly_data),
                        df_raw['company'].nunique(),
                        f"{df_raw['timestamp'].min()[:10]} to {df_raw['timestamp'].max()[:10]}",
                        formatted_month
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Data_Summary', index=False)
            
            # Clear cache directory
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            
            print(f"📦 Archived {len(monthly_data)} items to historical (JSON + Excel) and cleared cache")
            print(f"📊 Complete raw data saved to: {excel_filename}")
            return True
        else:
            print("⚠️ No data to archive")
            return False

# Initialize cache manager
cache_manager = CacheManager()
print("✅ Cache manager initialized")


# Cell 8: Enhanced Excel Report Generation with Visualizations - UPDATED
class ExcelReportGenerator:
    def __init__(self):
        self.stats = {}
        # Set professional styling
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1C32', '#6A8EAE']
    
    def generate_monthly_report(self, data, month_year):
        """Generate comprehensive Excel report for monthly reporting"""
        # Create historical data folder in same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        historical_dir = os.path.join(script_dir, 'Historical_Data_Market_Intelligence')
        os.makedirs(historical_dir, exist_ok=True)
        
        # Use Year-Month format for monthly reports
        formatted_month = datetime.now().strftime('%Y-%m')
        
        # Single monthly report
        filename = os.path.join(historical_dir, f"Market_Intelligence_Report_{formatted_month}.xlsx")
        
        # If file exists (shouldn't happen for monthly), add timestamp
        if os.path.exists(filename):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(historical_dir, f"Market_Intelligence_Report_{formatted_month}_{timestamp}.xlsx")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            try:
                # Tab 1: Executive Summary
                self.create_executive_summary(writer, data, formatted_month)
                
                # Tab 2: Competitive Intelligence Dashboard
                self.create_competitive_intelligence_dashboard(writer, data)
                
                # Tab 3: Enhanced Competitive Analysis
                self.create_competitive_analysis(writer, data)
                
                # Tab 4: Sentiment Analysis
                self.create_sentiment_analysis(writer, data)
                
                # Tab 5: NEW Visual Analytics Dashboard with 70% smaller charts
                self.create_enhanced_visual_analytics_dashboard(writer, data)
                
                # Tab 6: NEW Share of Voice Analysis
                self.create_share_of_voice_analysis(writer, data)
                
                # Tab 7: Raw Data
                self.create_raw_data_tab(writer, data)
                
                # Tab 8: Methodology & Confidence
                self.create_methodology_tab(writer)

            except Exception as e:
                print(f"⚠️ Report generation error: {e}")
                # Always create a fallback sheet so workbook saves
                pd.DataFrame([{"Error": str(e)}]).to_excel(writer, sheet_name="Error Log", index=False)
        
        print(f"📊 Enhanced Excel report generated: {filename}")
        return filename
    
    def create_executive_summary(self, writer, data, month_year):
        """Create executive summary tab"""
        summary_data = {
            'Metric': [
                'Report Period',
                'Total Companies Monitored',
                'Total Data Points Collected',
                'Overall Market Sentiment',
                'Sales Activity Detected',
                'Companies with Strategic Initiatives',
                'Social Media Coverage',
                'Professional Publication Mentions',
                'Top Performer (Sentiment)',
                'Most Active Company',
                'Data Confidence Score'
            ],
            'Value': [
                month_year,
                len(companies_config['companies']) if companies_config else 0,
                len(data),
                self.calculate_overall_sentiment(data),
                self.get_sales_activity_summary(data),
                self.get_strategic_initiatives_summary(data),
                self.get_social_media_coverage(data),
                self.get_publication_mentions_summary(data),
                self.get_top_performer(data),
                self.get_most_active_company(data),
                f"{self.calculate_confidence_score(data):.1%}"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        worksheet = writer.sheets['Executive Summary']
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 40

    def create_enhanced_visual_analytics_dashboard(self, writer, data):
        """Create enhanced visual analytics dashboard with 70% smaller charts"""
        if not data:
            pd.DataFrame([{"Info": "No data available for visual analytics"}]).to_excel(writer, sheet_name='Visual Analytics', index=False)
            return
        
        try:
            # Create all visualizations with smaller size
            self.create_sentiment_by_company_chart(data, writer)
            self.create_activity_distribution_by_company_chart(data, writer)
            self.create_professional_share_of_voice_chart(data, writer)
            self.create_social_media_share_of_voice_chart(data, writer)
            self.create_social_media_platform_distribution(data, writer)
            
        except Exception as e:
            print(f"⚠️ Enhanced visualization error: {e}")
            pd.DataFrame([{"Error": f"Enhanced visualization failed: {str(e)}"}]).to_excel(writer, sheet_name='Visual Analytics', index=False)

    def create_sentiment_by_company_chart(self, data, writer):
        """Create sentiment by company bar chart (70% smaller)"""
        fig, ax = plt.subplots(figsize=(8, 5))  # 70% smaller than original 12x8
        
        # Group data by company and sentiment
        company_sentiments = {}
        for item in data:
            company = item.get('company', 'Unknown')
            sentiment = item.get('sentiment', {}).get('overall_sentiment', 'neutral')
            
            if company not in company_sentiments:
                company_sentiments[company] = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            company_sentiments[company][sentiment] += 1
        
        # Prepare data for stacked bar chart
        companies = list(company_sentiments.keys())
        positive_counts = [company_sentiments[comp]['positive'] for comp in companies]
        neutral_counts = [company_sentiments[comp]['neutral'] for comp in companies]
        negative_counts = [company_sentiments[comp]['negative'] for comp in companies]
        
        # Create stacked bar chart
        bar_width = 0.6
        x_pos = np.arange(len(companies))
        
        ax.bar(x_pos, positive_counts, bar_width, label='Positive', color='#4CAF50', alpha=0.8)
        ax.bar(x_pos, neutral_counts, bar_width, bottom=positive_counts, label='Neutral', color='#FFC107', alpha=0.8)
        ax.bar(x_pos, negative_counts, bar_width, 
               bottom=[positive_counts[i] + neutral_counts[i] for i in range(len(companies))], 
               label='Negative', color='#F44336', alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Companies', fontweight='bold', fontsize=10)
        ax.set_ylabel('Sentiment Count', fontweight='bold', fontsize=10)
        ax.set_title('Sentiment Breakdown by Company', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(companies, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_plot_to_excel(fig, writer, 'Sentiment by Company')
        plt.close(fig)

    def create_activity_distribution_by_company_chart(self, data, writer):
        """Create activity distribution by company chart (70% smaller)"""
        fig, ax = plt.subplots(figsize=(8, 5))  # 70% smaller
        
        # Count activities by company
        company_activities = {}
        for item in data:
            company = item.get('company', 'Unknown')
            if company not in company_activities:
                company_activities[company] = Counter()
            
            # Count business activities
            activities = item.get('business_activities', {})
            for activity_type, items in activities.items():
                company_activities[company][activity_type] += len(items)
            
            # Count sales
            if item.get('sales_intelligence', {}).get('sale_detected', False):
                company_activities[company]['sales'] += 1
        
        # Prepare data for heatmap-style visualization
        activity_types = ['sales', 'events', 'launches', 'partnerships', 'expansion', 'awards']
        companies = list(company_activities.keys())
        
        heatmap_data = []
        for company in companies:
            row = [company_activities[company].get(activity, 0) for activity in activity_types]
            heatmap_data.append(row)
        
        # Create grouped bar chart
        x_pos = np.arange(len(companies))
        bar_width = 0.15
        
        for i, activity in enumerate(activity_types):
            activity_counts = [company_activities[company].get(activity, 0) for company in companies]
            ax.bar(x_pos + i * bar_width, activity_counts, bar_width, label=activity.title())
        
        ax.set_xlabel('Companies', fontweight='bold', fontsize=10)
        ax.set_ylabel('Activity Count', fontweight='bold', fontsize=10)
        ax.set_title('Activity Distribution by Company', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x_pos + bar_width * (len(activity_types) - 1) / 2)
        ax.set_xticklabels(companies, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_plot_to_excel(fig, writer, 'Activity by Company')
        plt.close(fig)

    def create_professional_share_of_voice_chart(self, data, writer):
        """Create professional publications share of voice chart"""
        fig, ax = plt.subplots(figsize=(6, 4))  # Compact size
        
        # Count professional publication mentions
        publication_mentions = Counter()
        for item in data:
            mentions = item.get('publication_mentions', [])
            for mention in mentions:
                publication = mention.get('publication', 'Unknown')
                publication_mentions[publication] += 1
        
        if publication_mentions:
            # Create pie chart
            labels = list(publication_mentions.keys())
            sizes = list(publication_mentions.values())
            
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90, textprops={'fontsize': 8})
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(7)
            
            ax.set_title('Share of Voice - Professional Publications', fontsize=10, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No professional publication data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        self.save_plot_to_excel(fig, writer, 'Professional Share of Voice')
        plt.close(fig)

    def create_social_media_share_of_voice_chart(self, data, writer):
        """Create social media share of voice chart"""
        fig, ax = plt.subplots(figsize=(6, 4))  # Compact size
        
        # Count social media activity by company
        social_activity = Counter()
        for item in data:
            company = item.get('company', 'Unknown')
            social_data = item.get('social_media_analysis', {})
            
            # Count active platforms
            active_platforms = sum(1 for platform_data in social_data.values() 
                                 if platform_data.get('activity_level') in ['medium', 'high'])
            social_activity[company] += active_platforms
        
        if social_activity:
            # Create bar chart
            companies = list(social_activity.keys())
            activity_counts = [social_activity[company] for company in companies]
            
            bars = ax.bar(companies, activity_counts, color='#2196F3', alpha=0.8)
            
            # Add value labels
            for bar, count in zip(bars, activity_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_xlabel('Companies', fontweight='bold', fontsize=9)
            ax.set_ylabel('Active Social Platforms', fontweight='bold', fontsize=9)
            ax.set_title('Share of Voice - Social Media', fontsize=10, fontweight='bold', pad=10)
            ax.set_xticklabels(companies, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No social media data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        self.save_plot_to_excel(fig, writer, 'Social Media Share of Voice')
        plt.close(fig)

    def create_social_media_platform_distribution(self, data, writer):
        """Create social media platform distribution chart"""
        fig, ax = plt.subplots(figsize=(6, 4))  # Compact size
        
        # Count platform usage
        platform_usage = Counter()
        for item in data:
            social_data = item.get('social_media_analysis', {})
            for platform, platform_data in social_data.items():
                if platform_data.get('activity_level') in ['medium', 'high']:
                    platform_usage[platform] += 1
        
        if platform_usage:
            # Create pie chart
            labels = [platform.title() for platform in platform_usage.keys()]
            sizes = list(platform_usage.values())
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90, textprops={'fontsize': 8})
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(7)
            
            ax.set_title('Social Media Platform Distribution', fontsize=10, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No social media platform data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        self.save_plot_to_excel(fig, writer, 'Social Media Platforms')
        plt.close(fig)

    def create_share_of_voice_analysis(self, writer, data):
        """Create comprehensive share of voice analysis with PIE CHARTS"""
        if not data:
            pd.DataFrame([{"Info": "No data available for share of voice analysis"}]).to_excel(
                writer, sheet_name='Share of Voice', index=False)
            return
        
        # Calculate professional publications share of voice
        professional_mentions = {}
        for item in data:
            company = item.get('company', 'Unknown')
            mentions = item.get('publication_mentions', [])
            professional_mentions[company] = professional_mentions.get(company, 0) + len(mentions)
        
        # Calculate social media share of voice
        social_media_mentions = {}
        for item in data:
            company = item.get('company', 'Unknown')
            social_data = item.get('social_media_analysis', {})
            active_platforms = sum(1 for platform_data in social_data.values() 
                                 if platform_data.get('activity_level') in ['medium', 'high'])
            social_media_mentions[company] = social_media_mentions.get(company, 0) + active_platforms
        
        # Create share of voice dataframe
        companies = list(set(list(professional_mentions.keys()) + list(social_media_mentions.keys())))
        
        share_data = []
        for company in companies:
            share_data.append({
                'Company': company,
                'Professional Publications Mentions': professional_mentions.get(company, 0),
                'Social Media Active Platforms': social_media_mentions.get(company, 0),
                'Total Share of Voice': professional_mentions.get(company, 0) + social_media_mentions.get(company, 0)
            })
        
        df_share = pd.DataFrame(share_data)
        df_share = df_share.sort_values('Total Share of Voice', ascending=False)
        
        # Add percentage columns
        total_professional = df_share['Professional Publications Mentions'].sum()
        total_social = df_share['Social Media Active Platforms'].sum()
        
        if total_professional > 0:
            df_share['Professional %'] = (df_share['Professional Publications Mentions'] / total_professional * 100).round(1)
        else:
            df_share['Professional %'] = 0
        
        if total_social > 0:
            df_share['Social Media %'] = (df_share['Social Media Active Platforms'] / total_social * 100).round(1)
        else:
            df_share['Social Media %'] = 0
        
        # Save the data table
        df_share.to_excel(writer, sheet_name='Share of Voice', startrow=0, index=False)
        
        # ⭐⭐⭐ NEW: CREATE BEAUTIFUL PIE CHARTS ⭐⭐⭐
        
        # Chart 1: Professional Publications Pie Chart
        if total_professional > 0 and len(professional_mentions) > 0:
            self.create_share_of_voice_pie_chart(
                professional_mentions, 
                "Share of Voice - Professional Publications",
                writer,
                'Professional Publications Pie Chart'
            )
        
        # Chart 2: Social Media Pie Chart  
        if total_social > 0 and len(social_media_mentions) > 0:
            self.create_share_of_voice_pie_chart(
                social_media_mentions,
                "Share of Voice - Social Media",
                writer,
                'Social Media Pie Chart'
            )
        
        # Chart 3: Total Share of Voice Pie Chart
        total_mentions = {}
        for company in companies:
            total_mentions[company] = professional_mentions.get(company, 0) + social_media_mentions.get(company, 0)
        
        if sum(total_mentions.values()) > 0 and len(total_mentions) > 0:
            self.create_share_of_voice_pie_chart(
                total_mentions,
                "Total Share of Voice",
                writer,
                'Total Share of Voice Pie Chart'
            )
        
        # Format the worksheet
        worksheet = writer.sheets['Share of Voice']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 25
        worksheet.column_dimensions['C'].width = 25
        worksheet.column_dimensions['D'].width = 20
        worksheet.column_dimensions['E'].width = 15
        worksheet.column_dimensions['F'].width = 15


    def create_share_of_voice_pie_chart(self, data_dict, title, writer, chart_name):
        """Create beautiful gradient pie chart for share of voice"""
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Prepare and sort data
        sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        companies = [item[0] for item in sorted_items if item[1] > 0]
        mentions = [item[1] for item in sorted_items if item[1] > 0]
    
        if not companies:
            return
    
        # Create gradient colors (blue gradient for professional, orange for social, green for total)
        if "Professional" in title:
            base_color = '#2E86AB'  # Professional blue
        elif "Social" in title:
            base_color = '#F18F01'  # Social orange
        else:
            base_color = '#4CAF50'  # Total green
        
        # Generate gradient colors
        colors = [plt.cm.Blues(0.3 + 0.7 * i/len(companies)) if "Professional" in title else
                  plt.cm.Oranges(0.3 + 0.7 * i/len(companies)) if "Social" in title else
                  plt.cm.Greens(0.3 + 0.7 * i/len(companies)) for i in range(len(companies))]
        
        # Explode the largest slice for emphasis
        explode = [0.1 if i == 0 else 0 for i in range(len(companies))]
        
        # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            mentions,
            labels=companies,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(mentions))})',
            colors=colors,
            explode=explode,
            startangle=90,
            shadow=True,
            textprops={'fontsize': 9, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'alpha': 0.9}
        )
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Beautiful title
        ax.set_title(f"📊 {title}\nTotal: {sum(mentions)} mentions", 
                    fontsize=16, fontweight='bold', pad=25, color='#333333')
        
        # Add elegant legend
        ax.legend(wedges, 
                 [f"{company}" for company in companies],
                 title="Companies",
                 loc="center left",
                 bbox_to_anchor=(1, 0.5),
                 fontsize=9,
                 title_fontsize=10)
        
        # Make it perfect
        ax.axis('equal')
        
        plt.tight_layout()
        self.save_plot_to_excel(fig, writer, chart_name)
        plt.close(fig)
    

    # Rest of the existing methods remain unchanged but are included for completeness
    def save_plot_to_excel(self, fig, writer, chart_name):
        """Save matplotlib figure to Excel worksheet"""
        # Convert plot to image
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Create worksheet for this chart
        worksheet = writer.book.create_sheet(chart_name)
        img = Image.open(img_buffer)
        
        # Insert image into worksheet
        from openpyxl.drawing.image import Image as XLImage
        xl_img = XLImage(img_buffer)
        xl_img.anchor = 'A1'
        worksheet.add_image(xl_img)
        
        # Set column width
        worksheet.column_dimensions['A'].width = 30

    def create_empty_plot(self, message, writer, chart_name):
        """Create placeholder for empty plots"""
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller empty plot
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes,
               fontsize=10, style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        self.save_plot_to_excel(fig, writer, chart_name)
        plt.close(fig)

    # Existing helper methods (shortened for brevity)
    def create_competitive_intelligence_dashboard(self, writer, data):
        """Create comprehensive competitive intelligence dashboard with evidence"""
        if not data:
            pd.DataFrame([{"Info": "No data available for dashboard"}]).to_excel(writer, sheet_name='Competitive Dashboard', index=False)
            return
        
        # Create competitive matrix with intelligence summaries and evidence
        matrix_data = []
        
        for company_data in data:
            company_name = company_data.get('company', 'Unknown')
            
            # Activity indicators
            sales_detected = company_data.get('sales_intelligence', {}).get('sale_detected', False)
            events_count = len(company_data.get('business_activities', {}).get('events', []))
            launches_count = len(company_data.get('business_activities', {}).get('launches', []))
            partnerships_count = len(company_data.get('business_activities', {}).get('partnerships', []))
            expansion_count = len(company_data.get('business_activities', {}).get('expansion', []))
            
            # NEW: Social media and publication metrics
            social_platforms = len(company_data.get('social_media_analysis', {}))
            publication_mentions = len(company_data.get('publication_mentions', []))
            
            # Generate intelligence summary
            intelligence_summary = company_data.get('intelligence_summary', 'No intelligence data')
            
            # Generate evidence summary
            evidence_summary = self.generate_evidence_summary(company_data)
            
            matrix_data.append({
                'Company': company_name,
                'Sales Activity': '✅' if sales_detected else '❌',
                'Events': events_count,
                'Product Launches': launches_count,
                'Partnerships': partnerships_count,
                'Expansion': expansion_count,
                'Social Platforms': social_platforms,
                'Publication Mentions': publication_mentions,
                'Business Intelligence Summary': intelligence_summary,
                'Evidence & Sources': evidence_summary
            })
        
        df_matrix = pd.DataFrame(matrix_data)
        df_matrix.to_excel(writer, sheet_name='Competitive Dashboard', index=False)
        
        # Format the dashboard
        worksheet = writer.sheets['Competitive Dashboard']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 15
        worksheet.column_dimensions['C'].width = 10
        worksheet.column_dimensions['D'].width = 15
        worksheet.column_dimensions['E'].width = 15
        worksheet.column_dimensions['F'].width = 12
        worksheet.column_dimensions['G'].width = 15
        worksheet.column_dimensions['H'].width = 18
        worksheet.column_dimensions['I'].width = 60
        worksheet.column_dimensions['J'].width = 80

    def generate_evidence_summary(self, company_data):
        """Generate evidence summary with specific sources"""
        evidence = company_data.get('evidence_sources', {})
        evidence_parts = []
        
        # Website source
        if evidence.get('website_url'):
            evidence_parts.append(f"🔗 {evidence['website_url']}")
        
        # Sales evidence
        sales_evidence = evidence.get('sales_evidence', [])
        if sales_evidence:
            evidence_parts.append(f"💰 Sales: '{sales_evidence[0][:50]}...'")
        
        # Activity evidence
        activity_evidence = evidence.get('activity_evidence', {})
        for activity_type, items in activity_evidence.items():
            if items:
                evidence_parts.append(f"🎯 {activity_type.title()}: '{items[0][:50]}...'")
        
        # Social media sources
        social_sources = evidence.get('social_media_sources', [])
        for source in social_sources[:2]:  # Limit to 2 platforms
            evidence_parts.append(f"📱 {source['platform']}: {source['activity_level']} activity")
        
        # Publication sources
        pub_sources = evidence.get('publication_sources', [])
        for source in pub_sources[:2]:  # Limit to 2 publications
            evidence_parts.append(f"📰 {source['publication']}")
        
        return " | ".join(evidence_parts) if evidence_parts else "No specific evidence captured"

    # Additional helper methods for executive summary
    def get_social_media_coverage(self, data):
        """Get social media coverage summary"""
        if not data:
            return "No data"
        
        companies_with_social = 0
        for item in data:
            social_data = item.get('social_media_analysis', {})
            if social_data:
                companies_with_social += 1
        
        return f"{companies_with_social}/{len(data)} companies"

    def get_publication_mentions_summary(self, data):
        """Get professional publication mentions summary"""
        if not data:
            return "No data"
        
        total_mentions = 0
        for item in data:
            total_mentions += len(item.get('publication_mentions', []))
        
        return f"{total_mentions} total mentions"

    # Rest of existing methods (create_competitive_analysis, create_sentiment_analysis, etc.)
    # ... [Previous methods remain unchanged]
    
    def create_competitive_analysis(self, writer, data):
        """Create competitive analysis with counts and ratio"""
        if not data:
            pd.DataFrame([{"Info": "No data"}]).to_excel(writer, sheet_name='Competitive Analysis', index=False)
            return

        df = pd.DataFrame(data)
        df['sentiment_overall'] = df['sentiment'].apply(
            lambda s: s.get('overall_sentiment', 'neutral') if isinstance(s, dict) else 'neutral'
        )
        company_stats = df.groupby('company').agg(
            total_mentions=('url', 'count'),
            positive_mentions=('sentiment_overall', lambda x: (x == 'positive').sum())
        )
        company_stats['positive_ratio'] = (company_stats['positive_mentions'] / company_stats['total_mentions']).round(3)
        company_stats = company_stats.sort_values('total_mentions', ascending=False)
        company_stats.to_excel(writer, sheet_name='Competitive Analysis')
        worksheet = writer.sheets['Competitive Analysis']
        worksheet.column_dimensions['A'].width = 25

    def create_sentiment_analysis(self, writer, data):
        """Create detailed sentiment analysis"""
        if not data:
            pd.DataFrame([{"Info": "No data"}]).to_excel(writer, sheet_name='Sentiment Analysis', index=False)
            return

        df = pd.DataFrame(data)
        df['sentiment_overall'] = df['sentiment'].apply(
            lambda s: s.get('overall_sentiment', 'neutral') if isinstance(s, dict) else 'neutral'
        )
        df['sentiment_confidence'] = df['sentiment'].apply(
            lambda s: s.get('confidence', 0) if isinstance(s, dict) else 0
        )

        sentiment_dist = df['sentiment_overall'].value_counts().rename_axis('Sentiment').reset_index(name='Count')
        sentiment_dist['Percentage'] = (sentiment_dist['Count'] / len(df) * 100).round(1)

        company_sentiment = df.groupby('company').agg(
            dominant_sentiment=('sentiment_overall', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral'),
            avg_confidence=('sentiment_confidence', 'mean'),
            mention_count=('url', 'count')
        ).round(3).sort_values('mention_count', ascending=False)

        sentiment_dist.to_excel(writer, sheet_name='Sentiment Analysis', startrow=0, index=False)
        company_sentiment.to_excel(writer, sheet_name='Sentiment Analysis', startrow=len(sentiment_dist) + 3, index=True)
        worksheet = writer.sheets['Sentiment Analysis']
        worksheet.column_dimensions['A'].width = 20
        
    def create_raw_data_tab(self, writer, data):
        """Create raw data tab for reference"""
        if not data:
            pd.DataFrame([{"Info": "No data"}]).to_excel(writer, sheet_name='Raw Data', index=False)
            return

        flat_data = []
        for item in data:
            flat_item = {
                'company': item.get('company', ''),
                'url': item.get('url', ''),
                'timestamp': item.get('timestamp', ''),
                'title': item.get('title', ''),
                'content_preview': item.get('content', '')[:100] + '...' if item.get('content') else '',
                'sentiment': item.get('sentiment', {}).get('overall_sentiment', '') if isinstance(item.get('sentiment', {}), dict) else '',
                'confidence': item.get('sentiment', {}).get('confidence', 0) if isinstance(item.get('sentiment', {}), dict) else 0,
                'key_phrases': ', '.join(item.get('key_phrases', [])[:3]) if isinstance(item.get('key_phrases'), list) else '',
                'sales_detected': item.get('sales_intelligence', {}).get('sale_detected', False),
                'business_activities': str([k for k, v in item.get('business_activities', {}).items() if v]),
                'social_media_platforms': ', '.join(item.get('social_media_analysis', {}).keys()),
                'publication_mentions': len(item.get('publication_mentions', []))
            }
            flat_data.append(flat_item)
        
        df_flat = pd.DataFrame(flat_data)
        df_flat.to_excel(writer, sheet_name='Raw Data', index=False)
        
        worksheet = writer.sheets['Raw Data']
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 30
        worksheet.column_dimensions['E'].width = 40
        
    def create_methodology_tab(self, writer):
        """Create methodology and confidence metrics tab"""
        methodology_data = {
            'Component': [
                'Data Collection',
                'Sentiment Analysis',
                'Sales Detection',
                'Business Activity Intelligence',
                'Social Media Scraping',
                'Professional Publications',
                'Visual Analytics',
                'Statistical Significance',
                'Confidence Scoring',
                'Share of Voice Metrics',
                'Scrapy Integration',
                'Selenium Integration'
            ],
            'Description': [
                'Automated web scraping every 2 days from configured sources + Selenium for JS-heavy sites',
                'Combined TextBlob + VADER analysis running locally',
                'Multi-pattern detection for sales, discounts, promotions',
                'Event, launch, partnership, expansion, and award detection',
                'Selenium-based scraping of Twitter, Instagram, LinkedIn, Facebook',
                'Scrapy-based monitoring of Bike Europe, Cycling Industry News, ECF, etc.',
                '6 professional visualizations with 70% smaller charts and clear labels',
                'Visualizations work with any sample size, no statistical filtering',
                'Based on model confidence and data completeness',
                'Separate metrics for Professional Publications vs Social Media',
                'Advanced Scrapy framework for efficient publication scraping',
                'Selenium WebDriver for JavaScript-rendered content and social media'
            ]
        }
        
        df_methodology = pd.DataFrame(methodology_data)
        df_methodology.to_excel(writer, sheet_name='Methodology', index=False)
        
        worksheet = writer.sheets['Methodology']
        worksheet.column_dimensions['A'].width = 25
        worksheet.column_dimensions['B'].width = 60

    # Enhanced helper methods for executive summary
    def calculate_overall_sentiment(self, data):
        if not data:
            return "Neutral"
        sentiments = [
            item.get('sentiment', {}).get('overall_sentiment', 'neutral')
            if isinstance(item.get('sentiment', {}), dict) else 'neutral'
            for item in data
        ]
        return max(set(sentiments), key=sentiments.count)
    
    def get_sales_activity_summary(self, data):
        """Count companies with detected sales"""
        if not data:
            return "No data"
        
        companies_with_sales = 0
        for item in data:
            sales_intel = item.get('sales_intelligence', {})
            if sales_intel.get('sale_detected', False):
                companies_with_sales += 1
        
        return f"{companies_with_sales}/{len(data)} companies"
    
    def get_strategic_initiatives_summary(self, data):
        """Count companies with strategic business activities"""
        if not data:
            return "No data"
        
        active_companies = 0
        for item in data:
            activities = item.get('business_activities', {})
            if any(len(items) > 0 for items in activities.values()):
                active_companies += 1
        
        return f"{active_companies}/{len(data)} companies"
    
    def get_top_performer(self, data):
        """Top company by average sentiment confidence"""
        if not data:
            return "None"
        df = pd.DataFrame(data)
        df['sentiment_confidence'] = df['sentiment'].apply(
            lambda s: s.get('confidence', 0) if isinstance(s, dict) else 0
        )
        if 'company' not in df.columns or df.empty:
            return "None"
        company_sentiment = df.groupby('company')['sentiment_confidence'].mean()
        return company_sentiment.idxmax() if not company_sentiment.empty else "None"
    
    def get_most_active_company(self, data):
        if not data:
            return "None"
        
        activity_scores = {}
        for item in data:
            company = item.get('company', '')
            activities = item.get('business_activities', {})
            score = sum(len(items) for items in activities.values())
            activity_scores[company] = score
        
        if activity_scores:
            return max(activity_scores, key=activity_scores.get)
        else:
            df = pd.DataFrame(data)
            if 'company' not in df.columns or df.empty:
                return "None"
            return df['company'].value_counts().idxmax()
    
    def calculate_confidence_score(self, data):
        if not data:
            return 0.0
        confidences = [
            item.get('sentiment', {}).get('confidence', 0)
            if isinstance(item.get('sentiment', {}), dict) else 0
            for item in data
        ]
        return (sum(confidences) / len(confidences)) if confidences else 0.0

# Initialize enhanced report generator
report_generator = ExcelReportGenerator()
print("✅ Enhanced report generator with Scrapy + Selenium visualizations initialized")


# Cell 9: Email Sending System (UNCHANGED)
class EmailSender:
    def __init__(self):
        self.config = load_email_config()
    
    def send_report(self, excel_file_path, month_year):
        """Send Excel report via email - THIS IS NOW MANDATORY"""
        if not self.config:
            print("❌ Email configuration not loaded - CANNOT SEND EMAIL")
            return False
            
        try:
            # Email setup
            msg = MIMEMultipart()
            msg['From'] = self.config.get('smtp_settings', {}).get('from_email', 'noreply@company.com')
            msg['To'] = self.config.get('primary_recipient', '')
            msg['Subject'] = f"Enhanced Market Intelligence Report - {month_year}"
            
            # Enhanced email body
            body = f"""
            Dear Recipient,
            
            Please find attached the enhanced monthly market intelligence report for {month_year}.
            
            This comprehensive report now includes:
            • Competitive intelligence dashboard with natural language summaries
            • Sales detection and business activity tracking
            • Strategic initiative monitoring (events, launches, partnerships, expansions)
            • Social media integration with platform-specific analysis
            • Professional publications monitoring with Scrapy
            • 70% smaller, more focused visualizations
            • True Share of Voice metrics (Professional vs Social Media)
            • Evidence sources and data transparency
            • Enhanced sentiment analysis with business context
            
            NEW VISUAL ANALYTICS:
            📊 Sentiment Breakdown by Company (Bar Chart)
            📈 Activity Distribution by Company (Individual Focus)
            📰 Share of Voice - Professional Publications  
            📱 Share of Voice - Social Media
            🔄 Social Media Platform Distribution
            🎯 True Share of Voice Analysis
            
            NEW TECHNOLOGIES:
            🚀 Scrapy for efficient professional publications scraping
            🌐 Selenium for JavaScript-heavy websites and social media
            📊 70% smaller, more focused charts in Excel
            
            Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            
            Key Features:
            🎯 Sales detection with confidence scoring
            📊 Business activity intelligence
            🏆 Strategic initiative tracking
            📈 Professional executive dashboard
            🎨 Beautiful, compact visualizations
            🔍 Transparent evidence sources
            📱 Social media integration
            📰 Professional publications tracking
            
            Best regards,
            Enhanced Market Intelligence System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach Excel file
            with open(excel_file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(excel_file_path)}'
            )
            msg.attach(part)
            
            # ✅ REAL SMTP SEND (Gmail) - MANDATORY
            self._send_via_smtp(msg)
            print(f"📧 ENHANCED VISUAL REPORT EMAIL SENT SUCCESSFULLY to {msg['To']}")
            return True
            
        except Exception as e:
            print(f"❌ FAILED TO SEND ENHANCED VISUAL REPORT EMAIL: {str(e)}")
            return False
    
    def _send_via_smtp(self, msg):
        """Send email via Gmail SMTP server"""
        smtp_config = self.config.get('smtp_settings', {})
        with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)


# Initialize email sender
email_sender = EmailSender()
print("✅ Enhanced email sender initialized")


# Cell 10: Enhanced Main Scheduler & Execution - UPDATED
class IntelligenceScheduler:
    def __init__(self):
        self.collector = DataCollector()
        self.cache_manager = CacheManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.report_generator = ExcelReportGenerator()
        self.email_sender = EmailSender()
    
    def run_data_collection(self):
        """Run every 2 days - enhanced data collection with business intelligence"""
        if not companies_config:
            print("❌ companies_config.json not loaded")
            return

        print(f"\n🔄 Starting ENHANCED data collection with Scrapy + Selenium: {datetime.now()}")
        
        all_collected_data = []
        
        # 1. Scrape company websites with enhanced intelligence
        for company_id, company_info in companies_config['companies'].items():
            print(f"\n🎯 Collecting enhanced intelligence for: {company_info['official_name']}")
            
            # Collect from main website with enhanced business intelligence
            website_data = self.collector.scrape_with_business_intelligence(
                company_info['website'], 
                company_info['official_name']
            )
            
            if website_data:
                all_collected_data.append(website_data)
                # Print intelligence summary
                summary = website_data.get('intelligence_summary', 'No summary')
                print(f"   📊 {summary}")
            else:
                print(f"   ❌ Failed to collect data for {company_info['official_name']}")
            
            # Polite delay
            time.sleep(2)
        
        # 2. NEW: Scrape professional publications using Scrapy
        print(f"\n📰 Starting professional publications scraping...")
        publication_data = self.collector.run_publications_scraping()
        if publication_data:
            all_collected_data.extend(publication_data)
            print(f"   ✅ Found {len(publication_data)} publication mentions")
        
        # Save to cache
        if all_collected_data:
            total_items = self.cache_manager.save_to_cache(all_collected_data)
            print(f"✅ ENHANCED collection complete: {total_items} total items in cache")
        else:
            print("⚠️ No data collected in this cycle")
    
    def run_monthly_analysis(self):
        """Run on 1st of each month - full enhanced analysis and reporting WITH MANDATORY EMAIL"""
        print(f"\n📈 Starting ENHANCED monthly analysis: {datetime.now()}")
        
        monthly_data = self.cache_manager.load_current_month_cache()
        
        if not monthly_data:
            print("❌ No data available for enhanced monthly analysis")
            return
        
        # Generate enhanced Excel report
        month_year = datetime.now().strftime('%m_%Y')
        excel_file = self.report_generator.generate_monthly_report(monthly_data, month_year)
        
        # ✅ MANDATORY: Send email with enhanced report - NO OPTION TO SKIP
        if os.path.exists(excel_file):
            email_success = self.email_sender.send_report(excel_file, month_year)
            
            if email_success:
                # Archive and clear cache (this now saves BOTH JSON and Excel historical data)
                self.cache_manager.archive_current_month()
                print("✅ ENHANCED monthly analysis complete - cache cleared for new month")
                print(f"📁 Enhanced visual report saved to: {excel_file}")
                print(f"📧 Enhanced visual report email sent successfully")
            else:
                print("❌ ENHANCED VISUAL REPORT EMAIL FAILED - Report generated but email not sent")
                print(f"📁 Enhanced visual report saved to: {excel_file}")
        else:
            print("❌ Enhanced visual Excel report generation failed")

# Initialize enhanced scheduler
scheduler = IntelligenceScheduler()
print("✅ Enhanced intelligence scheduler with Scrapy + Selenium initialized")


# Cell 11: Enhanced Manual Execution & Automation - UPDATED
def manual_data_collection():
    """Manually run enhanced data collection"""
    print("🚀 Starting ENHANCED manual data collection with Scrapy + Selenium...")
    scheduler.run_data_collection()

def manual_monthly_analysis():
    """Manually run enhanced monthly analysis WITH MANDATORY EMAIL"""
    print("🚀 Starting ENHANCED manual monthly analysis...")
    scheduler.run_monthly_analysis()

def generate_report_with_email():
    """Generate enhanced report and send email immediately"""
    print("🚀 GENERATING ENHANCED VISUAL REPORT AND SENDING EMAIL...")
    monthly_data = cache_manager.load_current_month_cache()
    
    if not monthly_data:
        print("❌ No data available for enhanced visual report generation")
        return
    
    # Generate enhanced Excel report
    month_year = datetime.now().strftime('%m_%Y')
    excel_file = report_generator.generate_monthly_report(monthly_data, month_year)
    
    # ✅ MANDATORY EMAIL SENDING
    if os.path.exists(excel_file):
        email_success = email_sender.send_report(excel_file, month_year)
        if email_success:
            print("✅ ENHANCED VISUAL report generated and email sent successfully!")
            print(f"📁 Enhanced visual report saved to: {excel_file}")
        else:
            print("❌ Enhanced visual report generated but email failed to send")
            print(f"📁 Enhanced visual report saved to: {excel_file}")
    else:
        print("❌ Enhanced visual report generation failed")

def test_business_intelligence():
    """Test the enhanced business intelligence on a single company"""
    print("🧪 Testing enhanced business intelligence detection with Scrapy + Selenium...")
    if companies_config and companies_config['companies']:
        test_company = list(companies_config['companies'].values())[0]
        test_data = collector.scrape_with_business_intelligence(
            test_company['website'], 
            test_company['official_name']
        )
        if test_data and 'sales_intelligence' in test_data:
            sales_info = test_data['sales_intelligence']
            activities = test_data['business_activities']
            evidence = test_data.get('evidence_sources', {})
            social_data = test_data.get('social_media_analysis', {})
            print(f"✅ Enhanced intelligence test for {test_company['official_name']}:")
            print(f"   Sale detected: {sales_info['sale_detected']} (confidence: {sales_info['confidence']}%)")
            print(f"   Sale keywords: {sales_info['sale_keywords_found']}")
            print(f"   Business activities: {[k for k, v in activities.items() if v]}")
            print(f"   Social media platforms: {list(social_data.keys())}")
            print(f"   Evidence sources: {evidence.get('website_url', 'No URL')}")
            print(f"   Intelligence summary: {test_data['intelligence_summary']}")
        else:
            print("❌ Enhanced business intelligence test failed")
    else:
        print("⚠️ No companies configured to test")

def test_selenium_scraping():
    """Test Selenium scraping functionality"""
    print("🧪 Testing Selenium scraping...")
    test_url = "https://example.com"
    selenium_data = collector.selenium_scraper.scrape_with_selenium(test_url)
    if selenium_data:
        print(f"✅ Selenium test successful: {len(selenium_data['content'])} characters extracted")
    else:
        print("❌ Selenium test failed")

def view_cache_contents():
    """View what's currently in cache"""
    cache_data = cache_manager.load_current_month_cache()
    print(f"\n📂 Current cache contents: {len(cache_data)} items")
    for item in cache_data[:3]:  # Show first 3 items
        company = item.get('company', '?')
        sales = item.get('sales_intelligence', {}).get('sale_detected', False)
        activities = [k for k, v in item.get('business_activities', {}).items() if v]
        social_platforms = list(item.get('social_media_analysis', {}).keys())
        publication_mentions = len(item.get('publication_mentions', []))
        print(f"  • {company} - Sales: {sales}, Activities: {activities}")
        print(f"    Social: {social_platforms}, Publications: {publication_mentions}")

def start_automated_scheduler():
    """Start the automated scheduler to run continuously"""
    print("🤖 Starting ENHANCED automated scheduler with Scrapy + Selenium...")
    
    # ✅ RELOAD config to get latest values
    current_config = load_analysis_period_config()
    scraping_days = current_config.get('scraping_frequency_days', 2)
    
    print(f"📅 ENHANCED data collection: Every {scraping_days} days")
    print("📅 ENHANCED monthly report: 1st of each month at 09:00")
    print("🎯 FEATURES: Sales detection, Business activity intelligence, Executive dashboard")
    print("📊 VISUALS: 70% smaller charts with Sentiment by Company, Activity Distribution")
    print("📱 SOCIAL: Twitter, Instagram, LinkedIn, Facebook integration")
    print("📰 PUBLICATIONS: Scrapy-based professional publications monitoring")
    print("🔍 EVIDENCE: Transparent data sources and detection evidence")
    print("🚀 TECHNOLOGY: Scrapy + Selenium for advanced scraping")
    print("📧 EMAIL IS MANDATORY FOR ALL ENHANCED VISUAL REPORTS")
    print("⏰ Running continuously... Press Ctrl+C to stop")
    
    # Schedule the tasks with dynamic frequency
    schedule.every(scraping_days).days.do(scheduler.run_data_collection)
    schedule.every().month.at("09:00").do(scheduler.run_monthly_analysis)
    
    # Run immediately once
    scheduler.run_data_collection()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


print("\n🎯 ENHANCED VISUAL INTELLIGENCE SYSTEM WITH SCRAPY + SELENIUM READY! YOU CAN NOW:")
print("1. Run manual_data_collection() - enhanced data collection with Scrapy + Selenium")
print("2. Run manual_monthly_analysis() - full enhanced analysis + ✅ MANDATORY EMAIL") 
print("3. Run generate_report_with_email() - quick enhanced visual report + ✅ MANDATORY EMAIL")
print("4. Run test_business_intelligence() - test sales & activity detection with evidence")
print("5. Run test_selenium_scraping() - test Selenium scraping functionality")
print("6. Run start_automated_scheduler() - continuous enhanced automation + ✅ MANDATORY EMAIL")
print("7. Run view_cache_contents() to see current enhanced data")
print("8. Configure email_config.json for actual email sending")

print("\n🚀 ENHANCED VISUAL MARKET INTELLIGENCE SYSTEM WITH SCRAPY + SELENIUM READY!")
print("✅ Sales detection & business activity tracking")
print("✅ Natural language intelligence summaries")  
print("✅ Evidence sources & transparent data tracking")
print("✅ 70% smaller, more focused visualizations")
print("✅ Executive dashboard with competitive intelligence")
print("✅ Social media integration (Twitter, Instagram, LinkedIn, Facebook)")
print("✅ Professional publications monitoring with Scrapy")
print("✅ Selenium for JavaScript-heavy websites")
print("✅ True Share of Voice metrics (Professional vs Social)")
print("✅ All original functionality preserved and enhanced!")

print("\n📊 NEW ENHANCED VISUAL ANALYTICS DASHBOARD INCLUDES:")
print("   • Sentiment Breakdown by Company (Bar Chart)")
print("   • Activity Distribution by Company (Individual Focus)")  
print("   • Share of Voice - Professional Publications (Pie Chart)")
print("   • Share of Voice - Social Media (Bar Chart)")
print("   • Social Media Platform Distribution (Pie Chart)")
print("   • True Share of Voice Analysis (Data Table)")

print("\n🚀 NEW TECHNOLOGIES INTEGRATED:")
print("   • Scrapy framework for efficient publication scraping")
print("   • Selenium WebDriver for JavaScript-heavy content")
print("   • Social media profile scraping and analysis")
print("   • Professional publication monitoring")
print("   • Enhanced evidence tracking with source URLs")

print("\n🎯 NEW STANDARDIZED MARKET REALITY DASHBOARD FUNCTIONS ADDED:")
print("   • analyze_strategic_postures() - Categorizes company market positions")
print("   • detect_live_market_movements() - Identifies active strategic moves")  
print("   • assess_competitive_threats() - Evaluates immediate competitive threats")
print("   • standardized_sentiment_analysis() - Consistent sentiment scoring")
print("   • standardized_share_of_voice() - Consistent market presence measurement")
print("   • generate_market_reality_report() - Comprehensive standardized report")
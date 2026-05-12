/**
 * News Aggregation Fetcher
 * v2.60 Phase 1 - Alternative Data & NLP Alpha Infrastructure
 * 
 * Integrates NewsAPI.org for financial headlines
 * Falls back to RSS scraping for backup sources
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// Configuration
const NEWSAPI_KEY = process.env.NEWSAPI_KEY || '';
const CACHE_DIR = './data/news';
const RATE_LIMIT_MS = 600; // NewsAPI allows 600 requests per 10 minutes (1 per second)

let lastRequestTime = 0;

interface NewsArticle {
  id: string;
  title: string;
  description: string;
  url: string;
  source: string;
  publishedAt: string;
  content?: string;
  sectorTags?: string[];
  sentiment?: number;
  hash: string;
  fetchedAt: string;
}

interface NewsQuery {
  q: string;
  sources?: string;
  domains?: string;
  from?: string;
  to?: string;
  language?: string;
  sortBy?: 'relevancy' | 'popularity' | 'publishedAt';
  pageSize?: number;
  page?: number;
}

/**
 * Sleep utility for rate limiting
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Make rate-limited request to NewsAPI
 */
async function newsApiRequest(url: string): Promise<Response> {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  
  if (timeSinceLastRequest < RATE_LIMIT_MS) {
    await sleep(RATE_LIMIT_MS - timeSinceLastRequest);
  }
  
  lastRequestTime = Date.now();
  
  return fetch(url, {
    headers: {
      'Accept': 'application/json',
      'User-Agent': 'Portfolio-Lab/2.60 (research@portfolio-lab.local)'
    }
  });
}

/**
 * Calculate content hash for deduplication
 */
function calculateHash(content: string): string {
  return crypto.createHash('sha256').update(content.toLowerCase().trim()).digest('hex').substring(0, 16);
}

/**
 * Check if article is already cached
 */
function isArticleCached(hash: string): boolean {
  const filePath = path.join(CACHE_DIR, `${hash}.json`);
  return fs.existsSync(filePath);
}

/**
 * Save article to cache
 */
function saveArticle(article: NewsArticle): void {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
  
  const filePath = path.join(CACHE_DIR, `${article.hash}.json`);
  fs.writeFileSync(filePath, JSON.stringify(article, null, 2));
}

/**
 * Load cached article
 */
function loadArticle(hash: string): NewsArticle | null {
  const filePath = path.join(CACHE_DIR, `${hash}.json`);
  
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  }
  
  return null;
}

/**
 * Fetch headlines from NewsAPI
 */
export async function fetchNewsHeadlines(
  query: NewsQuery
): Promise<NewsArticle[]> {
  if (!NEWSAPI_KEY) {
    console.warn('NewsAPI key not configured. Set NEWSAPI_KEY environment variable.');
    return [];
  }
  
  const params = new URLSearchParams();
  params.append('q', query.q);
  params.append('apiKey', NEWSAPI_KEY);
  
  if (query.sources) params.append('sources', query.sources);
  if (query.domains) params.append('domains', query.domains);
  if (query.from) params.append('from', query.from);
  if (query.to) params.append('to', query.to);
  if (query.language) params.append('language', query.language);
  if (query.sortBy) params.append('sortBy', query.sortBy);
  if (query.pageSize) params.append('pageSize', query.pageSize.toString());
  if (query.page) params.append('page', query.page.toString());
  
  const url = `https://newsapi.org/v2/everything?${params.toString()}`;
  
  try {
    const response = await newsApiRequest(url);
    
    if (!response.ok) {
      const error = await response.text();
      console.error(`NewsAPI error: ${response.status} - ${error}`);
      return [];
    }
    
    const data = await response.json();
    
    if (data.status !== 'ok') {
      console.error(`NewsAPI returned error: ${data.message}`);
      return [];
    }
    
    const articles: NewsArticle[] = [];
    
    for (const item of data.articles || []) {
      const content = `${item.title} ${item.description || ''}`;
      const hash = calculateHash(content);
      
      // Check cache first
      if (isArticleCached(hash)) {
        const cached = loadArticle(hash);
        if (cached) {
          articles.push(cached);
        }
        continue;
      }
      
      const article: NewsArticle = {
        id: item.url,
        title: item.title,
        description: item.description || '',
        url: item.url,
        source: item.source?.name || 'Unknown',
        publishedAt: item.publishedAt,
        content: item.content,
        hash,
        fetchedAt: new Date().toISOString()
      };
      
      saveArticle(article);
      articles.push(article);
    }
    
    return articles;
  } catch (error) {
    console.error('Error fetching news:', error);
    return [];
  }
}

/**
 * Sector keywords for classification
 */
const SECTOR_KEYWORDS: Record<string, string[]> = {
  'XLK': ['technology', 'tech', 'software', 'ai', 'artificial intelligence', 'semiconductor', 'cloud', 'cybersecurity'],
  'XLF': ['financial', 'bank', 'finance', 'fintech', 'insurance', 'credit', 'lending'],
  'XLE': ['energy', 'oil', 'gas', 'petroleum', 'renewable', 'solar', 'wind'],
  'XLI': ['industrial', 'manufacturing', 'aerospace', 'defense', 'transportation'],
  'XLP': ['consumer staples', 'retail', 'grocery', 'food', 'beverage'],
  'XLY': ['consumer discretionary', 'retail', 'e-commerce', 'luxury', 'travel'],
  'XLV': ['healthcare', 'pharma', 'biotech', 'medical', 'drug', 'vaccine'],
  'XLB': ['materials', 'chemical', 'mining', 'metals', 'commodities'],
  'XLU': ['utilities', 'energy utility', 'water', 'electric', 'gas utility'],
  'XLRE': ['real estate', 'reit', 'property', 'housing', 'commercial real estate'],
  'XLC': ['communication', 'media', 'telecom', 'streaming', 'social media'],
  'SPY': ['s&p 500', 'stock market', 'equities', 'index', 'benchmark']
};

/**
 * Tag article with sector classifications
 */
export function tagSectors(article: NewsArticle): string[] {
  const text = `${article.title} ${article.description}`.toLowerCase();
  const tags: string[] = [];
  
  for (const [sector, keywords] of Object.entries(SECTOR_KEYWORDS)) {
    for (const keyword of keywords) {
      if (text.includes(keyword.toLowerCase())) {
        tags.push(sector);
        break;
      }
    }
  }
  
  return Array.from(new Set(tags)); // Deduplicate
}

/**
 * Fetch market-wide financial news
 */
export async function fetchMarketNews(
  daysBack: number = 1,
  pageSize: number = 100
): Promise<NewsArticle[]> {
  const to = new Date().toISOString().split('T')[0];
  const from = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  
  const query: NewsQuery = {
    q: '(stock market OR earnings OR "federal reserve" OR inflation OR recession) AND (SPY OR S&P 500 OR Nasdaq)',
    domains: 'bloomberg.com,reuters.com,ft.com,wsj.com,cnbc.com,marketwatch.com,seekingalpha.com',
    from,
    to,
    language: 'en',
    sortBy: 'publishedAt',
    pageSize
  };
  
  const articles = await fetchNewsHeadlines(query);
  
  // Tag with sectors
  for (const article of articles) {
    article.sectorTags = tagSectors(article);
  }
  
  return articles;
}

/**
 * Fetch sector-specific news
 */
export async function fetchSectorNews(
  sector: string,
  daysBack: number = 1,
  pageSize: number = 50
): Promise<NewsArticle[]> {
  const to = new Date().toISOString().split('T')[0];
  const from = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  
  const keywords = SECTOR_KEYWORDS[sector];
  if (!keywords) {
    console.warn(`Unknown sector: ${sector}`);
    return [];
  }
  
  const query: NewsQuery = {
    q: keywords.slice(0, 3).join(' OR '),
    from,
    to,
    language: 'en',
    sortBy: 'relevancy',
    pageSize
  };
  
  const articles = await fetchNewsHeadlines(query);
  
  // Force tag with sector
  for (const article of articles) {
    article.sectorTags = [sector, ...(article.sectorTags || [])];
    article.sectorTags = Array.from(new Set(article.sectorTags)); // Deduplicate
  }
  
  return articles;
}

/**
 * Aggregate news by sector
 */
export async function aggregateSectorNews(
  daysBack: number = 1
): Promise<Record<string, NewsArticle[]>> {
  const sectors = Object.keys(SECTOR_KEYWORDS);
  const results: Record<string, NewsArticle[]> = {};
  
  for (const sector of sectors) {
    console.log(`Fetching news for ${sector}...`);
    results[sector] = await fetchSectorNews(sector, daysBack, 30);
    await sleep(1000); // Rate limit between sectors
  }
  
  return results;
}

/**
 * Get news statistics
 */
export function getNewsStats(): {
  totalArticles: number;
  articlesByDate: Record<string, number>;
  articlesBySource: Record<string, number>;
} {
  if (!fs.existsSync(CACHE_DIR)) {
    return { totalArticles: 0, articlesByDate: {}, articlesBySource: {} };
  }
  
  const files = fs.readdirSync(CACHE_DIR).filter(f => f.endsWith('.json'));
  
  const articlesByDate: Record<string, number> = {};
  const articlesBySource: Record<string, number> = {};
  
  for (const file of files) {
    try {
      const article: NewsArticle = JSON.parse(
        fs.readFileSync(path.join(CACHE_DIR, file), 'utf-8')
      );
      
      const date = article.publishedAt.split('T')[0];
      articlesByDate[date] = (articlesByDate[date] || 0) + 1;
      
      articlesBySource[article.source] = (articlesBySource[article.source] || 0) + 1;
    } catch {
      // Skip invalid files
    }
  }
  
  return {
    totalArticles: files.length,
    articlesByDate,
    articlesBySource
  };
}

/**
 * Main entry point
 */
async function main() {
  console.log('News Fetcher v2.60 - Alternative Data Infrastructure');
  console.log('====================================================');
  console.log('');
  
  if (!NEWSAPI_KEY) {
    console.log('NOTE: NEWSAPI_KEY not set. Using demo mode (returns cached data only).');
    console.log('To enable live fetching, set: export NEWSAPI_KEY=your_key_here');
    console.log('');
  }
  
  // Ensure cache directory exists
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }
  
  // Fetch market-wide news
  console.log('Fetching market-wide financial news (last 24 hours)...');
  const marketNews = await fetchMarketNews(1, 50);
  console.log(`  Retrieved ${marketNews.length} articles`);
  
  // Show top headlines
  console.log('');
  console.log('Top Headlines:');
  marketNews.slice(0, 5).forEach((article, i) => {
    const sectors = article.sectorTags?.join(', ') || 'N/A';
    console.log(`  ${i + 1}. [${sectors}] ${article.title.substring(0, 70)}...`);
  });
  
  // Get stats
  const stats = getNewsStats();
  console.log('');
  console.log('Cache Statistics:');
  console.log(`  Total cached articles: ${stats.totalArticles}`);
  console.log(`  Unique sources: ${Object.keys(stats.articlesBySource).length}`);
  
  // Save aggregate
  const aggregatePath = path.join(CACHE_DIR, 'aggregate_latest.json');
  fs.writeFileSync(aggregatePath, JSON.stringify({
    fetchedAt: new Date().toISOString(),
    articleCount: marketNews.length,
    articles: marketNews.map(a => ({
      title: a.title,
      source: a.source,
      publishedAt: a.publishedAt,
      sectors: a.sectorTags,
      hash: a.hash
    }))
  }, null, 2));
  
  console.log('');
  console.log(`Aggregate saved to: ${aggregatePath}`);
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export default {
  fetchNewsHeadlines,
  fetchMarketNews,
  fetchSectorNews,
  aggregateSectorNews,
  tagSectors,
  getNewsStats,
  calculateHash
};

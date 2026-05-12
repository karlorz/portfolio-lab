/**
 * SEC EDGAR Data Fetcher
 * v2.60 Phase 1 - Alternative Data & NLP Alpha Infrastructure
 * 
 * Fetches SEC filings (8-K, 10-Q, 10-K) for S&P 500 constituents
 * Rate limited to 10 requests/second per SEC guidelines
 * Extracts MD&A sections for sentiment analysis
 */

import * as fs from 'fs';
import * as path from 'path';

const EDGAR_BASE_URL = 'https://www.sec.gov/Archives/edgar/daily-index';
const EDGAR_BULK_URL = 'https://www.sec.gov/Archives/edgar/full-index';
const USER_AGENT = 'Portfolio-Lab Research (contact@portfolio-lab.local)';

// Rate limiting: 10 requests per second
const RATE_LIMIT_MS = 100;
let lastRequestTime = 0;

interface FilingEntry {
  cik: string;
  companyName: string;
  formType: string;
  dateFiled: string;
  accessionNumber: string;
  url: string;
}

interface FilingDocument {
  cik: string;
  accessionNumber: string;
  formType: string;
  filedDate: string;
  documentUrl: string;
  rawContent?: string;
  mdnaSection?: string;
  extractionTimestamp: string;
}

/**
 * Sleep utility for rate limiting
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Make rate-limited request to SEC EDGAR
 */
async function edgarRequest(url: string): Promise<Response> {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  
  if (timeSinceLastRequest < RATE_LIMIT_MS) {
    await sleep(RATE_LIMIT_MS - timeSinceLastRequest);
  }
  
  lastRequestTime = Date.now();
  
  return fetch(url, {
    headers: {
      'User-Agent': USER_AGENT,
      'Accept': 'application/json,text/html,application/xhtml+xml',
      'Accept-Encoding': 'gzip, deflate',
      'Host': 'www.sec.gov'
    }
  });
}

/**
 * Fetch company CIK from ticker symbol
 * Uses SEC's ticker-to-CIK mapping
 */
export async function fetchCIKFromTicker(ticker: string): Promise<string | null> {
  try {
    const response = await edgarRequest(
      `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${ticker}&type=&dateb=&owner=include&count=1&output=xml`
    );
    
    if (!response.ok) {
      console.error(`Failed to fetch CIK for ${ticker}: ${response.status}`);
      return null;
    }
    
    const xml = await response.text();
    const cikMatch = xml.match(/<CIK[^>]*>(\d+)<\/CIK>/);
    
    if (cikMatch) {
      return cikMatch[1].padStart(10, '0');
    }
    
    return null;
  } catch (error) {
    console.error(`Error fetching CIK for ${ticker}:`, error);
    return null;
  }
}

/**
 * Fetch recent filings for a CIK
 */
export async function fetchRecentFilings(
  cik: string,
  formTypes: string[] = ['8-K', '10-Q', '10-K'],
  limit: number = 10
): Promise<FilingEntry[]> {
  try {
    const typeParam = formTypes.map(t => `type=${t}`).join('&');
    const response = await edgarRequest(
      `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${cik}&${typeParam}&dateb=&owner=include&count=${limit}&output=atom`
    );
    
    if (!response.ok) {
      console.error(`Failed to fetch filings for CIK ${cik}: ${response.status}`);
      return [];
    }
    
    const xml = await response.text();
    const entries: FilingEntry[] = [];
    
    // Parse Atom feed entries - use split instead of matchAll for compatibility
    const entryParts = xml.split('<entry');
    
    for (let i = 1; i < entryParts.length; i++) {
      const entryXml = entryParts[i].split('</entry>')[0];
      
      const titleMatch = entryXml.match(/<title[^>]*>([\s\S]*?)<\/title>/);
      const formTypeMatch = entryXml.match(/<filing-type>([\s\S]*?)<\/filing-type>/);
      const dateMatch = entryXml.match(/<filing-date>([\s\S]*?)<\/filing-date>/);
      const accessionMatch = entryXml.match(/<accession-number>([\s\S]*?)<\/accession-number>/);
      const linkMatch = entryXml.match(/<filing-href>([\s\S]*?)<\/filing-href>/);
      
      if (accessionMatch && linkMatch) {
        entries.push({
          cik: cik.replace(/^0+/, ''),
          companyName: titleMatch ? titleMatch[1].trim() : 'Unknown',
          formType: formTypeMatch ? formTypeMatch[1] : 'Unknown',
          dateFiled: dateMatch ? dateMatch[1] : 'Unknown',
          accessionNumber: accessionMatch[1],
          url: linkMatch[1]
        });
      }
    }
    
    return entries;
  } catch (error) {
    console.error(`Error fetching filings for CIK ${cik}:`, error);
    return [];
  }
}

/**
 * Extract document URL from filing index page
 */
export async function extractDocumentUrl(filingUrl: string): Promise<string | null> {
  try {
    const response = await edgarRequest(filingUrl);
    
    if (!response.ok) {
      return null;
    }
    
    const html = await response.text();
    
    // Find the primary document (usually the first .htm or .txt link)
    const docMatch = html.match(/href="([^"]*\.(?:htm|html|txt))"/i);
    
    if (docMatch) {
      const baseUrl = filingUrl.substring(0, filingUrl.lastIndexOf('/') + 1);
      return baseUrl + docMatch[1];
    }
    
    return null;
  } catch (error) {
    console.error(`Error extracting document URL:`, error);
    return null;
  }
}

/**
 * Fetch and extract MD&A section from a filing
 */
export async function fetchFilingContent(documentUrl: string): Promise<Partial<FilingDocument> | null> {
  try {
    const response = await edgarRequest(documentUrl);
    
    if (!response.ok) {
      return null;
    }
    
    const content = await response.text();
    
    // Extract MD&A section
    const mdnaSection = extractMDNASection(content);
    
    return {
      documentUrl,
      rawContent: content,
      mdnaSection,
      extractionTimestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error(`Error fetching filing content:`, error);
    return null;
  }
}

/**
 * Extract Management Discussion & Analysis section from HTML/text
 */
function extractMDNASection(content: string): string | undefined {
  // Clean up HTML tags
  const textContent = content
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  
  // Look for MD&A section markers
  const mdnaPatterns = [
    /Item\s+7[.]?\s+Management[']?s\s+Discussion[\s\S]*?Item\s+8/,
    /MANAGEMENT['']?S\s+DISCUSSION\s+AND\s+ANALYSIS[\s\S]*?(?=QUANTITATIVE|FINANCIAL\s+STATEMENTS)/,
    /Item\s+2[.]?\s+Management[']?s\s+Discussion[\s\S]*?Item\s+3/,
  ];
  
  for (const pattern of mdnaPatterns) {
    const match = textContent.match(pattern);
    if (match) {
      // Limit to 50,000 characters to avoid processing huge documents
      return match[0].substring(0, 50000);
    }
  }
  
  // If no specific section found, return first 10,000 chars as fallback
  return textContent.substring(0, 10000) || undefined;
}

/**
 * Save filing data to disk
 */
export function saveFiling(
  cik: string,
  accessionNumber: string,
  filing: FilingDocument,
  basePath: string = './data/edgar'
): void {
  const dir = path.join(basePath, cik, accessionNumber.replace(/-/g, ''));
  
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  
  const filePath = path.join(dir, 'filing.json');
  fs.writeFileSync(filePath, JSON.stringify(filing, null, 2));
}

/**
 * Load existing filing data if available
 */
export function loadFiling(
  cik: string,
  accessionNumber: string,
  basePath: string = './data/edgar'
): FilingDocument | null {
  const filePath = path.join(
    basePath, 
    cik, 
    accessionNumber.replace(/-/g, ''),
    'filing.json'
  );
  
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  }
  
  return null;
}

/**
 * Check if filing is already cached
 */
export function isFilingCached(
  cik: string,
  accessionNumber: string,
  basePath: string = './data/edgar'
): boolean {
  const filePath = path.join(
    basePath,
    cik,
    accessionNumber.replace(/-/g, ''),
    'filing.json'
  );
  
  return fs.existsSync(filePath);
}

/**
 * Process a single ticker: fetch CIK, filings, and content
 */
export async function processTicker(
  ticker: string,
  formTypes: string[] = ['8-K', '10-Q', '10-K'],
  maxFilings: number = 5
): Promise<FilingDocument[]> {
  console.log(`Processing ${ticker}...`);
  
  const cik = await fetchCIKFromTicker(ticker);
  
  if (!cik) {
    console.error(`Could not find CIK for ${ticker}`);
    return [];
  }
  
  console.log(`  CIK: ${cik}`);
  
  const filings = await fetchRecentFilings(cik, formTypes, maxFilings);
  console.log(`  Found ${filings.length} filings`);
  
  const results: FilingDocument[] = [];
  
  for (const filing of filings) {
    // Check cache first
    if (isFilingCached(filing.cik, filing.accessionNumber)) {
      console.log(`  [Cached] ${filing.formType} - ${filing.dateFiled}`);
      const cached = loadFiling(filing.cik, filing.accessionNumber);
      if (cached) {
        results.push(cached);
      }
      continue;
    }
    
    console.log(`  [Fetching] ${filing.formType} - ${filing.dateFiled}`);
    
    const documentUrl = await extractDocumentUrl(filing.url);
    
    if (!documentUrl) {
      console.warn(`    Could not extract document URL`);
      continue;
    }
    
    const content = await fetchFilingContent(documentUrl);
    
    if (!content) {
      console.warn(`    Could not fetch filing content`);
      continue;
    }
    
    const filingDoc: FilingDocument = {
      cik: filing.cik,
      accessionNumber: filing.accessionNumber,
      formType: filing.formType,
      filedDate: filing.dateFiled,
      documentUrl,
      rawContent: content.rawContent,
      mdnaSection: content.mdnaSection,
      extractionTimestamp: content.extractionTimestamp || new Date().toISOString()
    };
    
    // Save to disk
    saveFiling(filing.cik, filing.accessionNumber, filingDoc);
    
    results.push(filingDoc);
    
    // Be extra conservative with rate limiting
    await sleep(150);
  }
  
  return results;
}

/**
 * Batch process multiple tickers
 */
export async function processTickers(
  tickers: string[],
  formTypes: string[] = ['8-K', '10-Q', '10-K'],
  maxFilings: number = 5
): Promise<Record<string, FilingDocument[]>> {
  const results: Record<string, FilingDocument[]> = {};
  
  for (const ticker of tickers) {
    try {
      results[ticker] = await processTicker(ticker, formTypes, maxFilings);
    } catch (error) {
      console.error(`Failed to process ${ticker}:`, error);
      results[ticker] = [];
    }
    
    // Add delay between tickers
    await sleep(500);
  }
  
  return results;
}

/**
 * Main entry point for CLI usage
 */
async function main() {
  // Ensure data directory exists
  if (!fs.existsSync('./data/edgar')) {
    fs.mkdirSync('./data/edgar', { recursive: true });
  }
  
  // Test with a few S&P 500 tickers
  const testTickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'];
  
  console.log('SEC EDGAR Fetcher v2.60 - Alternative Data Infrastructure');
  console.log('==========================================================');
  console.log(`Processing ${testTickers.length} tickers...`);
  console.log('');
  
  const startTime = Date.now();
  const results = await processTickers(testTickers, ['8-K', '10-Q'], 3);
  const elapsed = (Date.now() - startTime) / 1000;
  
  console.log('');
  console.log('==========================================================');
  console.log(`Completed in ${elapsed.toFixed(1)}s`);
  
  let totalFilings = 0;
  for (const [ticker, filings] of Object.entries(results)) {
    console.log(`  ${ticker}: ${filings.length} filings`);
    totalFilings += filings.length;
  }
  
  console.log(`Total filings fetched: ${totalFilings}`);
  console.log(`Data saved to: ./data/edgar/`);
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export default {
  fetchCIKFromTicker,
  fetchRecentFilings,
  extractDocumentUrl,
  fetchFilingContent,
  processTicker,
  processTickers,
  saveFiling,
  loadFiling,
  isFilingCached
};

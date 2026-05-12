/**
 * NLP Sentiment Analyzer
 * v2.60 Phase 2 - Alternative Data & NLP Alpha Infrastructure
 * 
 * Financial text sentiment analysis using FinBERT and local LLM models
 * Supports CPU and GPU (CUDA/MPS) inference
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync, spawn } from 'child_process';

// Configuration
const MODEL_CACHE_DIR = './models';
const DATA_DIR = './data';
const SIGNALS_DIR = './data/signals';

// FinBERT model identifier
const FINBERT_MODEL = 'yiyanghkust/finbert-tone';

interface SentimentResult {
  text: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  confidence: number;
  scores: {
    positive: number;
    neutral: number;
    negative: number;
  };
  processedAt: string;
}

interface EarningsSentiment {
  ticker: string;
  reportDate: string;
  quarter: string;
  segments: {
    ceo: SentimentResult;
    cfo: SentimentResult;
    qa: SentimentResult;
    overall: SentimentResult;
  };
  guidance: {
    direction: 'raised' | 'lowered' | 'maintained' | 'none';
    confidence: number;
    keywords: string[];
  };
  toneSurprise: number; // vs historical average
}

interface NewsSentimentAggregate {
  sector: string;
  timestamp: string;
  windowHours: number;
  articleCount: number;
  sentiment: SentimentResult;
  volumeScore: number; // 0-1 based on article count
  momentum: number; // Change from previous window
  topKeywords: string[];
}

interface CompositeSignal {
  timestamp: string;
  earnings: {
    weight: number;
    score: number;
    confidence: number;
  };
  news: {
    weight: number;
    score: number;
    confidence: number;
  };
  jobs: {
    weight: number;
    score: number;
    confidence: number;
  };
  social: {
    weight: number;
    score: number;
    confidence: number;
  };
  composite: {
    score: number; // -1 to 1
    regime: 'risk_on' | 'neutral' | 'risk_off';
    confidence: number;
    zScore: number;
  };
}

/**
 * Detect available compute device (CUDA, MPS, or CPU)
 */
function detectComputeDevice(): 'cuda' | 'mps' | 'cpu' {
  try {
    // Check for NVIDIA GPU
    const nvidiaSmi = execSync('nvidia-smi', { encoding: 'utf-8' });
    if (nvidiaSmi.includes('NVIDIA')) {
      console.log('✓ CUDA GPU detected');
      return 'cuda';
    }
  } catch {
    // nvidia-smi not available
  }

  // Check for Apple Silicon (MPS)
  if (process.platform === 'darwin' && process.arch === 'arm64') {
    console.log('✓ Apple Silicon detected, using MPS');
    return 'mps';
  }

  console.log('ℹ Using CPU inference');
  return 'cpu';
}

/**
 * Python script for FinBERT inference (embedded for portability)
 */
const FINBERT_SCRIPT = `
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT
try:
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    elif device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
    else:
        device = 'cpu'
    
    model.eval()
    
    # Read input
    texts = json.loads(sys.stdin.read())
    
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        labels = ['positive', 'neutral', 'negative']
        sentiment = labels[probs.argmax()]
        confidence = float(probs.max())
        
        results.append({
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {
                'positive': float(probs[0]),
                'neutral': float(probs[1]),
                'negative': float(probs[2])
            }
        })
    
    print(json.dumps(results))
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
`;

class SentimentAnalyzer {
  private device: 'cuda' | 'mps' | 'cpu';
  private pythonPath: string;
  private usePython: boolean;
  private mockMode: boolean;

  constructor(options: { mock?: boolean; pythonPath?: string } = {}) {
    this.device = detectComputeDevice();
    this.pythonPath = options.pythonPath || 'python3';
    this.mockMode = options.mock || false;
    this.usePython = !this.mockMode;

    // Ensure directories exist
    if (!fs.existsSync(MODEL_CACHE_DIR)) {
      fs.mkdirSync(MODEL_CACHE_DIR, { recursive: true });
    }
    if (!fs.existsSync(SIGNALS_DIR)) {
      fs.mkdirSync(SIGNALS_DIR, { recursive: true });
    }

    if (this.mockMode) {
      console.log('⚠ SentimentAnalyzer running in MOCK mode');
    } else {
      console.log(`✓ SentimentAnalyzer initialized (${this.device})`);
    }
  }

  /**
   * Analyze a batch of texts using FinBERT
   */
  async analyze(texts: string[]): Promise<SentimentResult[]> {
    if (this.mockMode) {
      return texts.map(text => this.mockAnalyze(text));
    }

    return this.pythonAnalyze(texts);
  }

  /**
   * Mock analysis for testing without Python
   */
  private mockAnalyze(text: string): SentimentResult {
    // Simple keyword-based sentiment for testing
    const positiveWords = ['growth', 'beat', 'strong', 'profit', 'expansion', 'raise', 'gain', 'improve', 'surge', 'outperform'];
    const negativeWords = ['miss', 'weak', 'loss', 'decline', 'cut', 'fall', 'drop', 'underperform', 'recession', 'layoff'];
    
    const lowerText = text.toLowerCase();
    const posCount = positiveWords.filter(w => lowerText.includes(w)).length;
    const negCount = negativeWords.filter(w => lowerText.includes(w)).length;
    
    let sentiment: 'positive' | 'neutral' | 'negative';
    let scores: { positive: number; neutral: number; negative: number };
    
    if (posCount > negCount) {
      sentiment = 'positive';
      scores = { positive: 0.6 + Math.random() * 0.3, neutral: 0.2, negative: 0.1 };
    } else if (negCount > posCount) {
      sentiment = 'negative';
      scores = { positive: 0.1, neutral: 0.2, negative: 0.6 + Math.random() * 0.3 };
    } else {
      sentiment = 'neutral';
      scores = { positive: 0.25, neutral: 0.5, negative: 0.25 };
    }
    
    const confidence = scores[sentiment];
    
    return {
      text: text.substring(0, 100),
      sentiment,
      confidence,
      scores,
      processedAt: new Date().toISOString()
    };
  }

  /**
   * Python-based FinBERT analysis
   */
  private async pythonAnalyze(texts: string[]): Promise<SentimentResult[]> {
    return new Promise((resolve, reject) => {
      // Write script to temp file
      const scriptPath = path.join(MODEL_CACHE_DIR, 'finbert_infer.py');
      fs.writeFileSync(scriptPath, FINBERT_SCRIPT);

      // Spawn Python process
      const python = spawn(this.pythonPath, [scriptPath, this.device]);
      
      let output = '';
      let error = '';

      python.stdout.on('data', (data) => {
        output += data.toString();
      });

      python.stderr.on('data', (data) => {
        error += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          console.warn(`Python process failed (code ${code}): ${error}`);
          // Fall back to mock analysis
          console.log('Falling back to mock analysis...');
          resolve(texts.map(text => this.mockAnalyze(text)));
          return;
        }

        try {
          const results = JSON.parse(output);
          const sentiments: SentimentResult[] = results.map((r: any, i: number) => ({
            text: texts[i].substring(0, 100),
            sentiment: r.sentiment,
            confidence: r.confidence,
            scores: r.scores,
            processedAt: new Date().toISOString()
          }));
          resolve(sentiments);
        } catch (e) {
          console.warn('Failed to parse Python output:', e);
          resolve(texts.map(text => this.mockAnalyze(text)));
        }
      });

      // Send input to Python
      python.stdin.write(JSON.stringify(texts));
      python.stdin.end();
    });
  }

  /**
   * Analyze earnings transcript
   */
  async analyzeEarnings(ticker: string, transcript: string, reportDate: string): Promise<EarningsSentiment> {
    // Segment transcript by speaker (simplified - would use more sophisticated parsing)
    const segments = this.parseTranscript(transcript);
    
    // Analyze each segment
    const ceoAnalysis = await this.analyze([segments.ceo]);
    const cfoAnalysis = await this.analyze([segments.cfo]);
    const qaAnalysis = await this.analyze([segments.qa]);
    const overallAnalysis = await this.analyze([transcript.substring(0, 2000)]);

    // Extract guidance information
    const guidance = this.extractGuidance(transcript);

    // Calculate tone surprise (would compare to historical in production)
    const toneSurprise = this.calculateToneSurprise(overallAnalysis[0]);

    const quarter = this.getQuarterFromDate(reportDate);

    return {
      ticker,
      reportDate,
      quarter,
      segments: {
        ceo: ceoAnalysis[0],
        cfo: cfoAnalysis[0],
        qa: qaAnalysis[0],
        overall: overallAnalysis[0]
      },
      guidance,
      toneSurprise
    };
  }

  /**
   * Parse transcript into segments
   */
  private parseTranscript(transcript: string): { ceo: string; cfo: string; qa: string } {
    // Simple heuristics - production would use speaker diarization
    const lines = transcript.split('\\n');
    let ceo = '';
    let cfo = '';
    let qa = '';

    let currentSection = 'ceo';

    for (const line of lines) {
      const lower = line.toLowerCase();
      if (lower.includes('cfo') || lower.includes('financial officer')) {
        currentSection = 'cfo';
      } else if (lower.includes('q&a') || lower.includes('question')) {
        currentSection = 'qa';
      }

      if (currentSection === 'ceo') ceo += line + ' ';
      else if (currentSection === 'cfo') cfo += line + ' ';
      else if (currentSection === 'qa') qa += line + ' ';
    }

    return { ceo, cfo, qa };
  }

  /**
   * Extract guidance information from transcript
   */
  private extractGuidance(transcript: string): { direction: 'raised' | 'lowered' | 'maintained' | 'none'; confidence: number; keywords: string[] } {
    const lower = transcript.toLowerCase();
    const keywords: string[] = [];

    const guidanceWords = ['guidance', 'outlook', 'forecast', 'expect', 'project', 'target'];
    for (const word of guidanceWords) {
      if (lower.includes(word)) keywords.push(word);
    }

    let direction: 'raised' | 'lowered' | 'maintained' | 'none' = 'none';
    let confidence = 0.5;

    if (keywords.length > 0) {
      if (lower.includes('raise') || lower.includes('increase') || lower.includes('raise guidance') || lower.includes('above')) {
        direction = 'raised';
        confidence = 0.7;
      } else if (lower.includes('lower') || lower.includes('decrease') || lower.includes('cut') || lower.includes('below') || lower.includes('reduce')) {
        direction = 'lowered';
        confidence = 0.7;
      } else if (lower.includes('reaffirm') || lower.includes('maintain') || lower.includes('in line') || lower.includes('unchanged')) {
        direction = 'maintained';
        confidence = 0.6;
      }
    }

    return { direction, confidence, keywords };
  }

  /**
   * Calculate tone surprise vs historical
   */
  private calculateToneSurprise(overall: SentimentResult): number {
    // Simplified - would use historical data in production
    const baseScore = overall.scores.positive - overall.scores.negative;
    return baseScore * overall.confidence;
  }

  /**
   * Get quarter from date
   */
  private getQuarterFromDate(dateStr: string): string {
    const date = new Date(dateStr);
    const month = date.getMonth();
    const year = date.getFullYear();
    const quarter = Math.floor(month / 3) + 1;
    return `Q${quarter} ${year}`;
  }

  /**
   * Aggregate news sentiment by sector
   */
  async aggregateNewsBySector(articles: { sector: string; title: string; description?: string }[], windowHours: number = 24): Promise<NewsSentimentAggregate[]> {
    const sectorGroups = new Map<string, typeof articles>();

    // Group by sector
    for (const article of articles) {
      if (!sectorGroups.has(article.sector)) {
        sectorGroups.set(article.sector, []);
      }
      sectorGroups.get(article.sector)!.push(article);
    }

    // Analyze each sector
    const results: NewsSentimentAggregate[] = [];

    for (const [sector, sectorArticles] of Array.from(sectorGroups.entries())) {
      const texts = sectorArticles.map(a => `${a.title}. ${a.description || ''}`);
      const sentiments = await this.analyze(texts);

      // Aggregate
      const avgScore = sentiments.reduce((sum, s) => sum + (s.scores.positive - s.scores.negative), 0) / sentiments.length;
      const avgConfidence = sentiments.reduce((sum, s) => sum + s.confidence, 0) / sentiments.length;

      // Extract keywords
      const keywords = this.extractKeywords(texts.join(' '));

      // Volume score (normalize to 0-1, assuming max 100 articles is high volume)
      const volumeScore = Math.min(sectorArticles.length / 100, 1);

      results.push({
        sector,
        timestamp: new Date().toISOString(),
        windowHours,
        articleCount: sectorArticles.length,
        sentiment: {
          text: sector,
          sentiment: avgScore > 0.1 ? 'positive' : avgScore < -0.1 ? 'negative' : 'neutral',
          confidence: avgConfidence,
          scores: {
            positive: Math.max(0, avgScore) + 0.33,
            neutral: 0.33,
            negative: Math.max(0, -avgScore) + 0.33
          },
          processedAt: new Date().toISOString()
        },
        volumeScore,
        momentum: 0, // Would calculate from historical
        topKeywords: keywords.slice(0, 10)
      });
    }

    return results;
  }

  /**
   * Extract keywords from text
   */
  private extractKeywords(text: string): string[] {
    const words = text.toLowerCase().split(/\\W+/);
    const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'to', 'of', 'and', 'in', 'for', 'on', 'at', 'by', 'with']);
    
    const wordCounts = new Map<string, number>();
    for (const word of words) {
      if (word.length > 3 && !stopWords.has(word)) {
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
      }
    }

    return Array.from(wordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([word]) => word);
  }

  /**
   * Calculate composite signal from all sources
   */
  calculateComposite(
    earningsSignals: EarningsSentiment[],
    newsSignals: NewsSentimentAggregate[],
    jobsScore: number,
    socialScore: number
  ): CompositeSignal {
    // Calculate earnings score (average tone surprise)
    const earningsScore = earningsSignals.length > 0
      ? earningsSignals.reduce((sum, e) => sum + e.segments.overall.scores.positive - e.segments.overall.scores.negative, 0) / earningsSignals.length
      : 0;
    const earningsConfidence = earningsSignals.length > 0
      ? earningsSignals.reduce((sum, e) => sum + e.segments.overall.confidence, 0) / earningsSignals.length
      : 0;

    // Calculate news score (volume-weighted)
    const totalNewsVolume = newsSignals.reduce((sum, n) => sum + n.articleCount, 0);
    const newsScore = totalNewsVolume > 0
      ? newsSignals.reduce((sum, n) => sum + (n.sentiment.scores.positive - n.sentiment.scores.negative) * n.volumeScore, 0) / newsSignals.length
      : 0;
    const newsConfidence = totalNewsVolume > 0
      ? newsSignals.reduce((sum, n) => sum + n.sentiment.confidence * n.volumeScore, 0) / totalNewsVolume
      : 0;

    // Weights per specification
    const weights = {
      earnings: 0.4,
      news: 0.3,
      jobs: 0.2,
      social: 0.1
    };

    // Composite score
    const compositeScore = 
      earningsScore * weights.earnings +
      newsScore * weights.news +
      jobsScore * weights.jobs +
      socialScore * weights.social;

    // Z-score calculation (simplified - would use historical std dev)
    const zScore = compositeScore / 0.3; // Assume std dev of 0.3

    // Determine regime
    let regime: 'risk_on' | 'neutral' | 'risk_off';
    if (zScore > 0.5) {
      regime = 'risk_on';
    } else if (zScore < -0.5) {
      regime = 'risk_off';
    } else {
      regime = 'neutral';
    }

    // Overall confidence
    const overallConfidence = 
      earningsConfidence * weights.earnings +
      newsConfidence * weights.news +
      0.5 * weights.jobs + // Jobs confidence assumed
      0.3 * weights.social; // Social confidence assumed lower

    return {
      timestamp: new Date().toISOString(),
      earnings: { weight: weights.earnings, score: earningsScore, confidence: earningsConfidence },
      news: { weight: weights.news, score: newsScore, confidence: newsConfidence },
      jobs: { weight: weights.jobs, score: jobsScore, confidence: 0.5 },
      social: { weight: weights.social, score: socialScore, confidence: 0.3 },
      composite: {
        score: compositeScore,
        regime,
        confidence: overallConfidence,
        zScore
      }
    };
  }

  /**
   * Save signal to disk
   */
  saveSignal(signal: CompositeSignal, filename: string = 'alternative_data_composite.json'): void {
    const filepath = path.join(SIGNALS_DIR, filename);
    fs.writeFileSync(filepath, JSON.stringify(signal, null, 2));
    console.log(`✓ Signal saved to ${filepath}`);
  }
}

export { SentimentAnalyzer, detectComputeDevice };
export type { SentimentResult, EarningsSentiment, NewsSentimentAggregate, CompositeSignal };

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];
  const mockMode = args.includes('--mock');

  const analyzer = new SentimentAnalyzer({ mock: mockMode });

  switch (command) {
    case 'test':
      // Test with sample texts
      const tests = [
        'We exceeded expectations with strong revenue growth of 15% quarter over quarter.',
        'The company reported disappointing earnings and lowered full-year guidance.',
        'Management maintained their outlook citing stable market conditions.'
      ];
      
      analyzer.analyze(tests).then(results => {
        console.log('\\nTest Results:');
        results.forEach((r, i) => {
          console.log(`\\n${i + 1}. "${tests[i].substring(0, 50)}..."`);
          console.log(`   Sentiment: ${r.sentiment} (${(r.confidence * 100).toFixed(1)}%)`);
          console.log(`   Scores: +${r.scores.positive.toFixed(2)} / ~${r.scores.neutral.toFixed(2)} / -${r.scores.negative.toFixed(2)}`);
        });
      });
      break;

    case 'earnings':
      // Test earnings analysis
      const sampleTranscript = `
        CEO: We are pleased to report another quarter of strong performance. 
        Revenue grew 12% year over year, exceeding our guidance of 8-10%.
        
        CFO: Gross margins expanded to 42%, driven by operational efficiencies.
        We are raising our full-year EPS guidance from $4.50 to $4.75.
        
        Q&A: Analyst: What about headwinds in Q3? 
        CEO: While we see some macro uncertainty, our backlog remains strong.
      `;
      
      analyzer.analyzeEarnings('AAPL', sampleTranscript, '2026-04-28').then(result => {
        console.log('\\nEarnings Analysis:');
        console.log(JSON.stringify(result, null, 2));
      });
      break;

    case 'help':
    default:
      console.log(`
Sentiment Analyzer - v2.60 NLP Alpha Infrastructure

Usage:
  npx ts-node src/nlp/sentiment_analyzer.ts <command> [options]

Commands:
  test          Run sentiment analysis tests
  earnings      Test earnings transcript analysis
  help          Show this help

Options:
  --mock        Run in mock mode (no Python/FinBERT required)
      `);
  }
}

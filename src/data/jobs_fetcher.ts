/**
 * Job Postings Data Fetcher
 * v2.60 Phase 1 - Alternative Data & NLP Alpha Infrastructure
 * 
 * Fetches job posting data as alternative economic indicator
 * Tracks hiring velocity by sector/company
 */

import * as fs from 'fs';
import * as path from 'path';

// Configuration
const JOBS_CACHE_DIR = './data/jobs';
const CACHE_DURATION_HOURS = 24;

interface JobPosting {
  id: string;
  title: string;
  company: string;
  sector: string;
  location: string;
  postedAt: string;
  description: string;
  salaryRange?: string;
  url: string;
  source: string;
  fetchedAt: string;
}

interface SectorHiringMetrics {
  sector: string;
  timestamp: string;
  windowDays: number;
  totalPostings: number;
  activeCompanies: number;
  avgSalary?: number;
  topRoles: { role: string; count: number }[];
  velocityScore: number; // 0-1, relative to historical
  growthRate: number; // MoM change
  confidence: number;
}

interface JobsSignal {
  timestamp: string;
  overallScore: number; // -1 to 1
  sectors: SectorHiringMetrics[];
  topHiringSectors: string[];
  laggingSectors: string[];
  laborMarketHealth: 'expanding' | 'stable' | 'contracting';
}

/**
 * Fetch job postings from multiple sources
 * Note: Production implementation would use actual APIs
 * This is a framework with mock/sample data for structure
 */
class JobsFetcher {
  private cacheDir: string;

  constructor(cacheDir: string = JOBS_CACHE_DIR) {
    this.cacheDir = cacheDir;
    
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }
  }

  /**
   * Fetch job postings from GitHub Jobs API (free, no auth)
   * https://jobs.github.com/api
   */
  async fetchGitHubJobs(keywords: string[] = ['finance', 'data', 'engineering']): Promise<JobPosting[]> {
    const allJobs: JobPosting[] = [];
    
    for (const keyword of keywords) {
      try {
        // Note: GitHub Jobs API deprecated; this is a pattern for other sources
        // Production would use Indeed, LinkedIn, or other commercial APIs
        console.log(`Fetching jobs for keyword: ${keyword}`);
        
        // Mock data for structure demonstration
        const mockJobs = this.generateMockJobs(keyword, 50);
        allJobs.push(...mockJobs);
        
      } catch (error) {
        console.warn(`Failed to fetch jobs for ${keyword}:`, error);
      }
    }

    return allJobs;
  }

  /**
   * Generate mock job data for testing and demonstration
   * Production would replace with actual API calls
   */
  private generateMockJobs(keyword: string, count: number): JobPosting[] {
    const sectors = ['Technology', 'Financials', 'Healthcare', 'Consumer', 'Energy', 'Industrials'];
    const companies: Record<string, string[]> = {
      'Technology': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Meta'],
      'Financials': ['JPMorgan', 'Bank of America', 'Goldman Sachs', 'BlackRock'],
      'Healthcare': ['UnitedHealth', 'Johnson & Johnson', 'Pfizer', 'Eli Lilly'],
      'Consumer': ['Walmart', 'Amazon', 'Home Depot', 'Costco'],
      'Energy': ['ExxonMobil', 'Chevron', 'NextEra', 'ConocoPhillips'],
      'Industrials': ['Boeing', 'Honeywell', 'Union Pacific', 'Caterpillar']
    };

    const roles = {
      'finance': ['Financial Analyst', 'Quant Researcher', 'Risk Manager', 'Portfolio Manager'],
      'data': ['Data Scientist', 'Data Engineer', 'ML Engineer', 'BI Analyst'],
      'engineering': ['Software Engineer', 'DevOps Engineer', 'SRE', 'Platform Engineer']
    };

    const jobs: JobPosting[] = [];
    
    for (let i = 0; i < count; i++) {
      const sector = sectors[Math.floor(Math.random() * sectors.length)];
      const companyList = companies[sector];
      const company = companyList[Math.floor(Math.random() * companyList.length)];
      const roleList = roles[keyword as keyof typeof roles] || roles['data'];
      const title = roleList[Math.floor(Math.random() * roleList.length)];
      
      jobs.push({
        id: `mock-${keyword}-${i}`,
        title,
        company,
        sector,
        location: ['New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Austin, TX', 'Remote'][Math.floor(Math.random() * 5)],
        postedAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
        description: `Seeking a ${title} to join our growing ${sector} team...`,
        url: '#',
        source: 'mock',
        fetchedAt: new Date().toISOString()
      });
    }

    return jobs;
  }

  /**
   * Save jobs to cache
   */
  saveJobs(jobs: JobPosting[], filename?: string): void {
    const timestamp = new Date().toISOString().split('T')[0];
    const filepath = path.join(this.cacheDir, filename || `jobs_${timestamp}.json`);
    
    // Load existing or create new
    let existing: JobPosting[] = [];
    if (fs.existsSync(filepath)) {
      existing = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
    }

    // Deduplicate by ID
    const seen = new Set(existing.map(j => j.id));
    const newJobs = jobs.filter(j => !seen.has(j.id));
    
    const combined = [...existing, ...newJobs];
    fs.writeFileSync(filepath, JSON.stringify(combined, null, 2));
    
    console.log(`✓ Saved ${newJobs.length} new jobs (${combined.length} total) to ${filepath}`);
  }

  /**
   * Load cached jobs
   */
  loadJobs(days: number = 30): JobPosting[] {
    const jobs: JobPosting[] = [];
    const cutoffDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

    const files = fs.readdirSync(this.cacheDir).filter(f => f.startsWith('jobs_') && f.endsWith('.json'));
    
    for (const file of files) {
      const filepath = path.join(this.cacheDir, file);
      const fileDate = new Date(file.replace('jobs_', '').replace('.json', ''));
      
      if (fileDate >= cutoffDate) {
        const data = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
        jobs.push(...data);
      }
    }

    return jobs;
  }

  /**
   * Calculate sector hiring metrics
   */
  calculateSectorMetrics(jobs: JobPosting[], windowDays: number = 30): SectorHiringMetrics[] {
    const cutoffDate = new Date(Date.now() - windowDays * 24 * 60 * 60 * 1000);
    const recentJobs = jobs.filter(j => new Date(j.postedAt) >= cutoffDate);

    // Group by sector
    const sectorGroups = new Map<string, JobPosting[]>();
    for (const job of recentJobs) {
      if (!sectorGroups.has(job.sector)) {
        sectorGroups.set(job.sector, []);
      }
      sectorGroups.get(job.sector)!.push(job);
    }

    const metrics: SectorHiringMetrics[] = [];

    for (const [sector, sectorJobs] of Array.from(sectorGroups.entries())) {
      const companies = new Set(sectorJobs.map(j => j.company));
      
      // Calculate top roles
      const roleCounts = new Map<string, number>();
      for (const job of sectorJobs) {
        roleCounts.set(job.title, (roleCounts.get(job.title) || 0) + 1);
      }
      
      const topRoles = Array.from(roleCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([role, count]) => ({ role, count }));

      // Velocity score (normalized posting count relative to baseline)
      // Simplified - would use historical data in production
      const baseline = 50; // Expected postings per sector
      const velocityScore = Math.min(sectorJobs.length / baseline, 1);

      // Growth rate (comparing to previous window)
      // Simplified - would compare to actual historical data
      const growthRate = (Math.random() - 0.5) * 0.4; // -20% to +20%

      metrics.push({
        sector,
        timestamp: new Date().toISOString(),
        windowDays,
        totalPostings: sectorJobs.length,
        activeCompanies: companies.size,
        topRoles,
        velocityScore,
        growthRate,
        confidence: Math.min(sectorJobs.length / 20, 0.95) // Higher confidence with more data
      });
    }

    return metrics.sort((a, b) => b.velocityScore - a.velocityScore);
  }

  /**
   * Generate overall jobs signal
   */
  generateSignal(metrics: SectorHiringMetrics[]): JobsSignal {
    // Calculate overall score based on weighted sector velocities
    const sectorWeights: Record<string, number> = {
      'Technology': 0.25,
      'Financials': 0.20,
      'Healthcare': 0.20,
      'Consumer': 0.15,
      'Industrials': 0.10,
      'Energy': 0.10
    };

    let weightedScore = 0;
    let totalWeight = 0;
    let growingSectors = 0;
    let contractingSectors = 0;

    for (const m of metrics) {
      const weight = sectorWeights[m.sector] || 0.1;
      weightedScore += m.velocityScore * weight * (1 + m.growthRate);
      totalWeight += weight;
      
      if (m.growthRate > 0.05) growingSectors++;
      if (m.growthRate < -0.05) contractingSectors++;
    }

    const normalizedScore = weightedScore / totalWeight;
    
    // Map to -1 to 1 scale
    const overallScore = (normalizedScore - 0.5) * 2;

    // Determine labor market health
    let laborMarketHealth: 'expanding' | 'stable' | 'contracting';
    if (overallScore > 0.3) {
      laborMarketHealth = 'expanding';
    } else if (overallScore < -0.3) {
      laborMarketHealth = 'contracting';
    } else {
      laborMarketHealth = 'stable';
    }

    // Top and lagging sectors
    const sorted = [...metrics].sort((a, b) => b.velocityScore - a.velocityScore);
    const topHiringSectors = sorted.slice(0, 3).map(m => m.sector);
    const laggingSectors = sorted.slice(-3).map(m => m.sector);

    return {
      timestamp: new Date().toISOString(),
      overallScore,
      sectors: metrics,
      topHiringSectors,
      laggingSectors,
      laborMarketHealth
    };
  }

  /**
   * Main fetch and process pipeline
   */
  async fetchAndProcess(): Promise<JobsSignal> {
    // Check cache freshness
    const cacheFile = path.join(this.cacheDir, 'latest_signal.json');
    if (fs.existsSync(cacheFile)) {
      const cache = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));
      const cacheTime = new Date(cache.timestamp);
      const hoursOld = (Date.now() - cacheTime.getTime()) / (1000 * 60 * 60);
      
      if (hoursOld < CACHE_DURATION_HOURS) {
        console.log(`✓ Using cached jobs signal (${hoursOld.toFixed(1)}h old)`);
        return cache;
      }
    }

    // Fetch new data
    console.log('Fetching fresh jobs data...');
    const jobs = await this.fetchGitHubJobs();
    this.saveJobs(jobs);

    // Calculate metrics
    const metrics = this.calculateSectorMetrics(jobs);
    const signal = this.generateSignal(metrics);

    // Save signal
    fs.writeFileSync(cacheFile, JSON.stringify(signal, null, 2));
    
    console.log(`✓ Jobs signal generated: ${signal.laborMarketHealth} (score: ${signal.overallScore.toFixed(2)})`);
    
    return signal;
  }
}

export { JobsFetcher };
export type { JobPosting, SectorHiringMetrics, JobsSignal };

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  const fetcher = new JobsFetcher();

  switch (command) {
    case 'fetch':
      fetcher.fetchAndProcess().then(signal => {
        console.log('\\nJobs Signal:');
        console.log(`  Labor Market: ${signal.laborMarketHealth.toUpperCase()}`);
        console.log(`  Overall Score: ${signal.overallScore.toFixed(3)}`);
        console.log(`  Top Sectors: ${signal.topHiringSectors.join(', ')}`);
        console.log(`  Lagging: ${signal.laggingSectors.join(', ')}`);
        console.log('\\nSector Details:');
        signal.sectors.forEach(s => {
          console.log(`  ${s.sector}: ${s.totalPostings} postings, velocity=${s.velocityScore.toFixed(2)}, growth=${(s.growthRate * 100).toFixed(1)}%`);
        });
      });
      break;

    case 'mock':
      // Generate and save mock data
      const mockJobs = fetcher['generateMockJobs']('data', 200);
      fetcher.saveJobs(mockJobs, 'mock_sample.json');
      const metrics = fetcher.calculateSectorMetrics(mockJobs);
      const signal = fetcher.generateSignal(metrics);
      
      console.log('\\nMock Jobs Signal:');
      console.log(JSON.stringify(signal, null, 2));
      break;

    case 'help':
    default:
      console.log(`
Jobs Fetcher - v2.60 Alternative Data Infrastructure

Usage:
  npx ts-node src/data/jobs_fetcher.ts <command>

Commands:
  fetch         Fetch and process job postings
  mock          Generate mock data for testing
  help          Show this help
      `);
  }
}

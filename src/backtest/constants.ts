export const PROJECT_RISK_FREE_RATE_PERCENT = 4.5;
export const DEFAULT_RISK_FREE_RATE = PROJECT_RISK_FREE_RATE_PERCENT / 100;

type EnvironmentSource = Record<string, string | undefined>;

function readProcessEnv(): EnvironmentSource | undefined {
  return (globalThis as typeof globalThis & {
    process?: { env?: EnvironmentSource };
  }).process?.env;
}

function parseRiskFreeRatePercent(rawValue: string | undefined): number | null {
  if (rawValue === undefined || rawValue.trim() === '') {
    return null;
  }

  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? parsed : null;
}

export function getDefaultRiskFreeRate(env: EnvironmentSource | undefined = readProcessEnv()): number {
  const overridePercent = parseRiskFreeRatePercent(env?.RISK_FREE_RATE);
  return (overridePercent ?? PROJECT_RISK_FREE_RATE_PERCENT) / 100;
}

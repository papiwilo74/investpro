/// <reference types="vite/client" />

declare global {
  interface Window {
    Components: {
      toast: (message: string, type: 'info' | 'success' | 'error' | 'warning') => void;
      skeleton: (type: string) => React.ReactNode;
      metricCard: (label: string, value: string, subtitle?: string, color?: string) => React.ReactNode;
      verdictCard: (verdict: string, color: string, advice: string) => React.ReactNode;
      advisorStatCard: (label: string, value: string, subtitle: string, color: string) => React.ReactNode;
      signalBadge: (action: string, strength: number, reason: string) => React.ReactNode;
      tradesTable: (trades: any[]) => React.ReactNode;
      progressBar: (label: string, value: number, color: string) => React.ReactNode;
      featureImportanceChart: (importances: Record<string, number>) => React.ReactNode;
      portfolioWeightsChart: (weights: Record<string, number>) => React.ReactNode;
      validationReport: (report: any) => React.ReactNode;
      geneticReport: (result: any) => React.ReactNode;
      frontierScatterPlot: (frontier: any[], maxSharpe: any, minVol: any) => HTMLCanvasElement | null;
      newsCard: (title: string, publisher: string, link: string, time: string, sentiment: string) => React.ReactNode;
    };
    api: any;
    __gaPoll: NodeJS.Timeout | undefined;
  }
}

export {};

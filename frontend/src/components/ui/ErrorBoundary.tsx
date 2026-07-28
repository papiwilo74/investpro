import { Component, type ReactNode, type ErrorInfo } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error, info);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="glass-card text-center py-12">
          <div className="text-4xl mb-3">⚠</div>
          <h3 className="text-lg font-bold text-slate-800 dark:text-teal-50/90 mb-2">Algo salió mal</h3>
          <p className="text-sm text-slate-500 dark:text-teal-300/60 mb-4">{this.state.error?.message || 'Error inesperado'}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="btn-primary px-6 py-2 text-sm"
          >
            Reintentar
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

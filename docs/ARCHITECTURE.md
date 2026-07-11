# Arquitectura del Sistema

## Diagrama de Alto Nivel

```mermaid
graph TB
    subgraph "Data Layer"
        YF[yfinance Provider]
        AL[Alpaca Provider]
        PL[Polygon Provider]
        CM[CacheManager<br/>SQLite + Parquet]
        SA[SplitAdjuster]
        DM[DataManager<br/>Failover Chain]
        YF --> CM
        AL --> CM
        PL --> CM
        CM --> SA
        YF --> DM
        AL --> DM
        PL --> DM
    end

    subgraph "Feature Engineering"
        FG[FeatureGenerator<br/>20+ features técnicas]
        PFG[PanelFeatureGenerator<br/>Cross-sectional features<br/>Ticker embeddings]
    end

    subgraph "Modelos ML"
        XGB[XGBoost<br/>Por ticker]
        LGB[LightGBM<br/>Panel Model]
        NN[Neural Brain<br/>PyTorch]
        RL[RL Agent<br/>Q-Learning]
        LSTM[LSTM<br/>Secuencial]
        ENS[AdaptiveEnsemble<br/>Pesos dinámicos]
        MG[Model Gate<br/>Fail-closed]
        CC[Champion/Challenger<br/>Promoción]
        XGB --> ENS
        LGB --> ENS
        NN --> ENS
        RL --> ENS
        LSTM --> ENS
        ENS --> MG
        MG --> CC
    end

    subgraph "Backtesting"
        BE[BacktestEngine<br/>OHLCV simulation]
        BBE[BotBacktestEngine<br/>Full pipeline]
        WFO[Walk-Forward<br/>Optimization]
        MC[Monte Carlo<br/>Simulation]
        FV[FullValidation<br/>Pipeline unificado]
        BE --> BBE
        BBE --> WFO
        WFO --> MC
        MC --> FV
    end

    subgraph "API Layer"
        FA[FastAPI<br/>REST endpoints]
        AU[Auth JWT]
        MID[Middleware<br/>CORS, RateLimit, Security]
        PR[Prometheus<br/>/metrics]
        HL[HealthCheck<br/>/health]
        FA --> AU
        FA --> MID
        FA --> PR
        FA --> HL
    end

    subgraph "Frontend"
        RE[React SPA<br/>Vite + TypeScript]
        RT[React Router<br/>Dashboard/Trading/ML]
        RQ[React Query<br/>API Cache]
        LC[Lightweight Charts<br/>TradingView]
        RE --> RT
        RE --> RQ
        RE --> LC
    end

    subgraph "Observabilidad"
        SJ[structlog<br/>JSON logs]
        PM[Prometheus<br/>Métricas]
        GR[Grafana<br/>Dashboards]
        AL[Alerts<br/>Telegram/Discord]
        SJ --> PM
        PM --> GR
        GR --> AL
    end

    subgraph "Despliegue"
        CI[GitHub Actions<br/>CI/CD]
        DK[Docker]
        RD[Render]
        CI --> DK
        DK --> RD
    end

    Data --> Data Layer
    Data Layer --> Feature Engineering
    Feature Engineering --> Modelos ML
    Modelos ML --> Backtesting
    Backtesting --> API Layer
    API Layer --> Frontend
    API Layer --> Observabilidad
```

## Flujo de una Predicción

```mermaid
sequenceDiagram
    participant U as Usuario/Frontend
    participant API as FastAPI
    participant DM as DataManager
    participant CM as CacheManager
    participant YF as yfinance Provider
    participant FG as FeatureGenerator
    participant ENS as AdaptiveEnsemble
    participant MG as ModelGate

    U->>API: GET /api/ml/predict?ticker=AAPL
    API->>DM: get_data("AAPL", "1y", "1d")
    DM->>CM: get("AAPL_1y_1d")
    alt Cache Fresh
        CM-->>DM: DataFrame
    else Cache Expired
        DM->>YF: fetch("AAPL")
        YF-->>DM: DataFrame
        DM->>CM: set("AAPL_1y_1d", df)
    end
    DM-->>API: OHLCV DataFrame
    API->>FG: build_features(df)
    FG-->>API: X (feature matrix)
    loop For each model type
        API->>ENS: predict(regime, xgboost_signal, ...)
        ENS->>MG: is_approved(model)
        MG-->>ENS: True/False
    end
    ENS-->>API: EnsembleResult{score, confidence, weights}
    API-->>U: {direction, probability, confidence}
```

## Flujo de Datos

```mermaid
flowchart LR
    subgraph "Fuentes"
        YF[yfinance]
        AL[Alpaca]
        PL[Polygon]
    end
    subgraph "Caché"
        CACHE[(SQLite Index<br/>+ Parquet)]
    end
    subgraph "Procesamiento"
        TECH[Técnicos<br/>RSI, MACD, BB...]
        MACRO[Macro<br/>SPY, VIX]
        CS[Cross-sectional<br/>Ranks, Momentum]
        EMB[Ticker Embeddings<br/>Frequency, Sector, Target]
    end
    subgraph "Modelos"
        XGB[XGBoost]
        LGB[LightGBM Panel]
        NN[Neural Brain]
        ENS[Ensemble]
    end
    subgraph "Salida"
        PRED[Predicción]
        TRADE[Trade Signal]
        LOG[Log + Metrics]
    end

    YF --> CACHE
    AL --> CACHE
    PL --> CACHE
    CACHE --> TECH
    CACHE --> MACRO
    TECH --> CS
    CS --> EMB
    TECH --> XGB
    TECH --> LGB
    TECH --> NN
    EMB --> LGB
    MACRO --> ENS
    XGB --> ENS
    LGB --> ENS
    NN --> ENS
    ENS --> PRED
    PRED --> TRADE
    TRADE --> LOG
```

## Arquitectura de Configuración

```mermaid
classDiagram
    class Settings {
        +ENV: str
        +ALPACA_API_KEY: str
        +ALPACA_SECRET_KEY: str
        +TELEGRAM_BOT_TOKEN: str
        +DATA_PROVIDER: str
        +DATA_CACHE_TTL_HOURS: float
        +MAX_DAILY_LOSS_PCT: float
        +LEVERAGE_ENABLED: bool
        +feature_flags: FeatureFlags
        +validate(): list[str]
    }
    class FeatureFlags {
        +live_trading: bool
        +paper_trading: bool
        +train_ml: bool
        +alerts: bool
        +to_dict(): dict
    }
    class BrokerConfig {
        +api_key: str
        +paper: bool
        +leverage_enabled: bool
    }
    class RiskConfig {
        +max_daily_loss_pct: float
        +max_sector_exposure_pct: float
        +circuit_breaker_minutes: int
    }
    Settings --> FeatureFlags
    Settings --> BrokerConfig
    Settings --> RiskConfig
```

## Arquitectura de CI/CD

```mermaid
flowchart LR
    PUSH[Push a master] --> CI[GitHub Actions CI]
    CI --> LINT[Ruff + mypy]
    CI --> TEST[pytest]
    CI --> SEC[Bandit]
    LINT --> DEPLOY[Deploy to Render]
    TEST --> DEPLOY
    SEC --> DEPLOY
    DEPLOY --> TG[Telegram Notificación]
```

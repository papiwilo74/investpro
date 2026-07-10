# Machine Learning module
# Imports de torch son opcionales para que el bot web funcione sin torch instalado
from .features import FeatureGenerator
from .train import ModelTrainer

try:
    from .neural_brain import NeuralTradingBrain, NeuralTrainer, extract_features, train_from_backtest
except (ImportError, ModuleNotFoundError):
    NeuralTradingBrain = None
    NeuralTrainer = None
    train_from_backtest = None
    extract_features = None

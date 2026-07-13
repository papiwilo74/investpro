from abc import ABC, abstractmethod


class BaseBrokerClient(ABC):
    """Interfaz abstracta para desacoplar el bot de brokers específicos (Alpaca, IBKR, etc.)"""

    @abstractmethod
    def is_connected(self) -> bool:
        """Verifica la conexión con el servidor del broker."""
        pass

    @abstractmethod
    def get_account_summary(self) -> dict:
        """Retorna un resumen de la cuenta (fondos, buying power, PnL diario)."""
        pass

    @abstractmethod
    def get_positions(self) -> list:
        """Retorna la lista de posiciones abiertas actualmente."""
        pass

    @abstractmethod
    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        """Ejecuta una orden a precio de mercado."""
        pass

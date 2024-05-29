from typing import Optional
from datetime import datetime
from Models.Broker import Broker

class Order:
    def __init__(self, time: datetime, price: float, order_type: str, quantity: int, symbol: str, market: str, broker: Broker, order_id: Optional[str] = None):
        self.time = time
        self.price = price
        self.order_type = order_type  # "buy" or "sell"
        self.quantity = quantity
        self.symbol = symbol
        self.broker = broker
        self.market = market
        self.order_id = order_id
        self.status = "pending"  # "pending", "filled", "cancelled", "rejected"
        self.filled_quantity = 0
        self.filled_price = 0.0
        self.fees = 0.0

    def get_amount(self) -> float:
        return self.quantity * self.price

    def get_fees(self) -> float:
        self.fees = self.broker.calculate_fees(self)
        return self.fees

    def set_order_id(self, order_id: str):
        self.order_id = order_id

    def set_status(self, status: str):
        self.status = status

    def set_filled_quantity(self, filled_quantity: int):
        self.filled_quantity = filled_quantity

    def set_filled_price(self, filled_price: float):
        self.filled_price = filled_price

    def is_filled(self) -> bool:
        return self.status == "filled"

    def is_cancelled(self) -> bool:
        return self.status == "cancelled"

    def is_rejected(self) -> bool:
        return self.status == "rejected"
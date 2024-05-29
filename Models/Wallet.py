from typing import List
from Models.Order import Order
from Models.Broker import Broker

class Wallet:
    def __init__(self, initial_amount: float, broker: Broker, orders: List[Order] = None):
        self.initial_amount = initial_amount
        self.current_amount = initial_amount
        self.broker = broker
        self.orders = orders or []
        self.total_profit = 0
        self.total_fees = 0

    def add_order(self, order: Order):
        self.orders.append(order)
        self.current_amount -= order.amount * order.price
        self.current_amount -= self.broker.calculate_fees(order)
        self.total_fees += self.broker.calculate_fees(order)

    def close_order(self, order: Order, close_price: float):
        profit = (close_price - order.price) * order.amount
        self.total_profit += profit
        self.current_amount += order.amount * close_price

    def get_income(self) -> tuple:
        return self.total_profit - self.total_fees, self.total_profit, self.total_fees

    def get_current_amount(self) -> float:
        return self.current_amount

    def get_total_invested(self) -> float:
        return sum(order.amount * order.price for order in self.orders)

    def get_total_fees(self) -> float:
        return self.total_fees

    def get_total_profit(self) -> float:
        return self.total_profit

    def get_orders(self) -> List[Order]:
        return self.orders
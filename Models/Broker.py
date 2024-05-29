from typing import Dict
from Models.Order import Order
import requests

class Broker:
    def __init__(self, name: str, api_key: str, market_fees: Dict[str, Dict[str, float]]):
        self.name = name
        self.api_key = api_key
        self.market_fees = market_fees
        self.base_url = "https://api.trading212.com"

    def calculate_fees(self, order: Order) -> float:
        fees = self.market_fees[order.market]["fees_fixed"]
        fees += self.market_fees[order.market]["fees_percent"] * order.get_amount()
        fees += self.market_fees[order.market]["fees_per_share"] * order.quantity
        return fees

    def place_order(self, order: Order) -> bool:
        endpoint = f"{self.base_url}/v1/orders"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "instrument": order.symbol,
            "side": order.order_type,
            "quantity": order.quantity,
            "price": order.price,
            "orderType": "MARKET"
        }

        response = requests.post(endpoint, headers=headers, json=data)

        if response.status_code == 200:
            order_data = response.json()
            order.set_order_id(order_data["orderId"])
            order.set_status("pending")
            return True
        else:
            order.set_status("rejected")
            return False

    def cancel_order(self, order: Order) -> bool:
        endpoint = f"{self.base_url}/v1/orders/{order.order_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.delete(endpoint, headers=headers)

        if response.status_code == 200:
            order.set_status("cancelled")
            return True
        else:
            return False

    def get_order_status(self, order: Order) -> str:
        endpoint = f"{self.base_url}/v1/orders/{order.order_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(endpoint, headers=headers)

        if response.status_code == 200:
            order_data = response.json()
            order.set_status(order_data["status"])
            order.set_filled_quantity(order_data["filledQuantity"])
            order.set_filled_price(order_data["filledPrice"])
            return order.status
        else:
            return "unknown"
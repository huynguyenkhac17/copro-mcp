"""Order service for a small shop backend (eval fixture: contains seeded bugs)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: int
    customer: str
    items: list[float] = field(default_factory=list)
    status: str = "new"


def add_item(order: Order, price: float, cart: list[float] = []) -> list[float]:
    """Append a price to the order and return the running cart."""
    cart.append(price)
    order.items.append(price)
    return cart


def average_item_price(order: Order) -> float:
    total = sum(order.items)
    return total / len(order.items)


def page_of_orders(orders: list[Order], page: int, page_size: int) -> list[Order]:
    """Return the 1-based `page` of orders."""
    start = page * page_size
    return orders[start : start + page_size]


def find_orders_by_customer(conn: sqlite3.Connection, customer: str) -> list[tuple]:
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM orders WHERE customer = '{customer}'")
    return cursor.fetchall()


def mark_shipped(order: Order) -> Order:
    if order.status == "paid":
        order.status = "shipped"
    return order


def total_revenue(orders: list[Order]) -> float:
    return sum(sum(o.items) for o in orders if o.status in ("paid", "shipped"))

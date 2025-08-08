"""
Utility functions for the Avellaneda-Stoikov Market Maker.
"""
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional, List, Tuple
import math
import numpy as np


def round_to_tick_size(price: Decimal, tick_size: Decimal, side: str = 'buy') -> Decimal:
    """
    Round price to nearest tick size.
    For buy orders, round down (more conservative).
    For sell orders, round up (more conservative).
    """
    if side == 'buy':
        # Round down for buy orders
        return (price / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
    else:
        # Round up for sell orders
        return (price / tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * tick_size


def round_to_step_size(quantity: Decimal, step_size: Decimal) -> Decimal:
    """Round quantity to nearest step size (always round down to be safe)."""
    return (quantity / step_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_size


def calculate_ewma_volatility(returns: List[float], lambda_param: float = 0.94) -> float:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    
    Args:
        returns: List of log returns
        lambda_param: Decay factor (typically 0.94)
    
    Returns:
        Annualized volatility estimate
    """
    if len(returns) < 2:
        return 0.02  # Default 2% volatility
    
    # Calculate EWMA variance
    weights = np.array([(1 - lambda_param) * lambda_param ** i 
                       for i in range(len(returns) - 1, -1, -1)])
    weights = weights / weights.sum()
    
    variance = np.sum(weights * np.array(returns) ** 2)
    volatility = math.sqrt(variance)
    
    # Annualize (assuming minute-level returns)
    # 252 trading days * 24 hours * 60 minutes
    annualization_factor = math.sqrt(252 * 24 * 60)
    
    return volatility * annualization_factor


def calculate_order_imbalance(bids: List[Tuple[float, float]], 
                             asks: List[Tuple[float, float]], 
                             depth: int = 5) -> float:
    """
    Calculate order book imbalance.
    
    Returns:
        Imbalance between -1 (all ask pressure) and 1 (all bid pressure)
    """
    bid_volume = sum(float(size) for price, size in bids[:depth])
    ask_volume = sum(float(size) for price, size in asks[:depth])
    
    if bid_volume + ask_volume == 0:
        return 0.0
    
    return (bid_volume - ask_volume) / (bid_volume + ask_volume)


def calculate_spread_bps(bid: Decimal, ask: Decimal) -> Decimal:
    """Calculate spread in basis points."""
    if bid == 0:
        return Decimal('0')
    
    mid = (bid + ask) / 2
    spread = ask - bid
    
    return (spread / mid) * Decimal('10000')


def estimate_fill_probability(distance_from_mid: Decimal, 
                             volatility: Decimal, 
                             time_horizon: float = 1.0) -> float:
    """
    Estimate probability of fill based on distance from mid and volatility.
    Uses simplified normal distribution assumption.
    
    Args:
        distance_from_mid: How far the order is from mid price (as fraction)
        volatility: Current volatility estimate
        time_horizon: Time horizon in seconds
    
    Returns:
        Probability between 0 and 1
    """
    if volatility == 0:
        return 0.5
    
    # Normalize distance by expected price movement
    sigma_t = float(volatility) * math.sqrt(time_horizon / 86400)  # Daily vol to time horizon
    z_score = float(distance_from_mid) / sigma_t
    
    # Use normal CDF approximation
    # For buy orders below mid, higher distance = lower fill probability
    # For sell orders above mid, higher distance = lower fill probability
    from scipy.stats import norm
    
    return 1 - norm.cdf(abs(z_score))


def calculate_inventory_risk(position: Decimal, 
                            volatility: Decimal, 
                            time_horizon: float = 3600) -> Decimal:
    """
    Calculate inventory risk (simplified VaR).
    
    Args:
        position: Current position
        volatility: Volatility estimate
        time_horizon: Time horizon in seconds
    
    Returns:
        Risk in price units
    """
    # Simple VaR calculation
    sigma_t = volatility * Decimal(str(math.sqrt(time_horizon / 86400)))
    
    # 95% confidence level (1.65 standard deviations)
    var_95 = abs(position) * sigma_t * Decimal('1.65')
    
    return var_95


def format_metrics(metrics: dict) -> str:
    """Format metrics dictionary for logging."""
    lines = []
    lines.append("=== Market Maker Metrics ===")
    lines.append(f"Position: {metrics.get('position', 0):.4f}")
    lines.append(f"PnL: ${metrics.get('realized_pnl', 0):.2f}")
    lines.append(f"Volume: {metrics.get('total_volume', 0):.4f}")
    lines.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
    lines.append(f"Fill Rate: {metrics.get('fill_rate', 0):.1%}")
    lines.append(f"Orders: {metrics.get('order_count', 0)} (Fills: {metrics.get('fill_count', 0)}, Rejects: {metrics.get('reject_count', 0)})")
    
    return "\n".join(lines)


class RollingMetrics:
    """Track rolling metrics over time windows."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize with window size."""
        self.window_size = window_size
        self.fills = []
        self.orders = []
        self.pnl_history = []
        
    def add_fill(self, timestamp: float, price: float, quantity: float, side: str):
        """Add a fill to history."""
        self.fills.append({
            'timestamp': timestamp,
            'price': price,
            'quantity': quantity,
            'side': side
        })
        
        # Trim to window size
        if len(self.fills) > self.window_size:
            self.fills = self.fills[-self.window_size:]
    
    def add_order(self, timestamp: float):
        """Add an order to history."""
        self.orders.append(timestamp)
        
        # Trim to window size
        if len(self.orders) > self.window_size:
            self.orders = self.orders[-self.window_size:]
    
    def get_fill_rate(self, time_window: float = 3600) -> float:
        """Get fill rate over time window (in seconds)."""
        if not self.orders:
            return 0.0
        
        now = max(self.orders[-1], max(f['timestamp'] for f in self.fills) if self.fills else 0)
        cutoff = now - time_window
        
        recent_orders = sum(1 for t in self.orders if t >= cutoff)
        recent_fills = sum(1 for f in self.fills if f['timestamp'] >= cutoff)
        
        if recent_orders == 0:
            return 0.0
        
        return recent_fills / recent_orders
    
    def get_avg_spread_captured(self) -> Optional[float]:
        """Calculate average spread captured (sell price - buy price)."""
        if len(self.fills) < 2:
            return None
        
        buy_fills = [f for f in self.fills if f['side'] == 'buy']
        sell_fills = [f for f in self.fills if f['side'] == 'sell']
        
        if not buy_fills or not sell_fills:
            return None
        
        avg_buy = sum(f['price'] * f['quantity'] for f in buy_fills) / sum(f['quantity'] for f in buy_fills)
        avg_sell = sum(f['price'] * f['quantity'] for f in sell_fills) / sum(f['quantity'] for f in sell_fills)
        
        return avg_sell - avg_buy
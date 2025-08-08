"""
Lean Avellaneda-Stoikov Market Maker
Combines optimal pricing theory with practical patterns from Rust MM implementation.
"""
import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """Decision with traceable reasons (from Rust MM pattern)."""
    action: str  # "send", "cancel", "nothing"
    reasons: List[str]
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    order_id: Optional[str] = None


@dataclass
class OpenOrder:
    """Track an open order."""
    order_id: str
    side: str  # "buy" or "sell"
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    cancel_pending: bool = False


@dataclass
class MarketState:
    """Current market and strategy state."""
    # Position
    position: Decimal = Decimal('0')
    
    # Open orders
    buy_order: Optional[OpenOrder] = None
    sell_order: Optional[OpenOrder] = None
    
    # Lockout timestamps (from Rust MM)
    last_fill_time: Dict[str, datetime] = field(default_factory=lambda: {
        'buy': datetime.min, 'sell': datetime.min
    })
    last_order_time: Dict[str, datetime] = field(default_factory=lambda: {
        'buy': datetime.min, 'sell': datetime.min
    })
    last_reject_time: Dict[str, datetime] = field(default_factory=lambda: {
        'buy': datetime.min, 'sell': datetime.min
    })
    
    # Market data
    last_mid_price: Optional[Decimal] = None
    volatility: Decimal = Decimal('0.02')  # 2% default
    
    # Metrics
    realized_pnl: Decimal = Decimal('0')
    total_volume: Decimal = Decimal('0')
    fill_count: int = 0
    order_count: int = 0
    reject_count: int = 0


class AvellanedaStoikovMM:
    """
    Lean Avellaneda-Stoikov Market Maker.
    Combines AS optimal pricing with Rust MM's practical patterns.
    """
    
    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config
        self.state = MarketState()
        self.state_lock = asyncio.Lock()
        
        # AS model parameters
        self.gamma = Decimal(str(config.get('gamma', 0.1)))
        self.initial_volatility = Decimal(str(config.get('volatility', 0.02)))
        self.max_position = Decimal(str(config.get('max_position', 1.0)))
        self.min_position = Decimal(str(config.get('min_position', -1.0)))
        self.order_size = Decimal(str(config.get('order_size', 0.1)))
        
        # Lockout durations (from Rust MM)
        self.fill_lockout = timedelta(milliseconds=config.get('fill_lockout_ms', 1000))
        self.order_lockout = timedelta(milliseconds=config.get('order_lockout_ms', 500))
        self.reject_lockout = timedelta(milliseconds=config.get('reject_lockout_ms', 2000))
        
        # Tolerance for order updates (from Rust MM)
        self.tolerance_frac = Decimal(str(config.get('tolerance_frac', 0.001)))
        
        # Trading session parameters
        self.session_length = config.get('session_hours', 24) * 3600  # Convert to seconds
        self.session_start = datetime.now()
        
        # Price history for volatility estimation
        self.price_history = deque(maxlen=100)
        
        # Order management
        self.next_order_id = 1
        
        logger.info(f"Initialized AS MM with gamma={self.gamma}, max_pos={self.max_position}")
    
    async def update_state(self, update_fn):
        """
        Update state with locking (from Rust pattern).
        All state modifications go through here.
        """
        async with self.state_lock:
            update_fn(self.state)
    
    def calculate_reservation_price(self, mid_price: Decimal) -> Decimal:
        """
        Calculate Avellaneda-Stoikov reservation price.
        r(t) = s(t) - q * γ * σ² * (T - t)
        """
        time_left = max(1, self.session_length - (datetime.now() - self.session_start).seconds)
        time_factor = Decimal(str(time_left / self.session_length))
        
        inventory_adjustment = (
            self.state.position * 
            self.gamma * 
            self.state.volatility ** 2 * 
            time_factor
        )
        
        return mid_price - inventory_adjustment
    
    def calculate_optimal_spread(self) -> Decimal:
        """
        Calculate Avellaneda-Stoikov optimal spread.
        δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
        """
        time_left = max(1, self.session_length - (datetime.now() - self.session_start).seconds)
        time_factor = Decimal(str(time_left / self.session_length))
        
        # Simplified k (order arrival rate) - could be estimated from data
        k = Decimal('1.5')
        
        spread = (
            self.gamma * self.state.volatility ** 2 * time_factor +
            (Decimal('2') / self.gamma) * Decimal(str(math.log(1 + float(self.gamma / k))))
        )
        
        # Apply minimum spread
        min_spread = Decimal('0.0001')  # 1 basis point minimum
        return max(spread, min_spread)
    
    def check_lockouts(self, side: str) -> List[str]:
        """Check if we're within any lockout period (from Rust MM)."""
        reasons = []
        now = datetime.now()
        
        if now < self.state.last_fill_time[side] + self.fill_lockout:
            reasons.append("fill_lockout")
        
        if now < self.state.last_order_time[side] + self.order_lockout:
            reasons.append("order_lockout")
        
        if now < self.state.last_reject_time[side] + self.reject_lockout:
            reasons.append("reject_lockout")
        
        return reasons
    
    def check_position_limits(self, side: str, quantity: Decimal) -> List[str]:
        """Check if order would breach position limits."""
        reasons = []
        
        if side == 'buy':
            new_position = self.state.position + quantity
            if new_position > self.max_position:
                reasons.append("max_position")
        else:
            new_position = self.state.position - quantity
            if new_position < self.min_position:
                reasons.append("min_position")
        
        return reasons
    
    def should_cancel_order(self, order: OpenOrder, target_price: Decimal) -> bool:
        """
        Check if order should be cancelled based on tolerance (from Rust MM).
        Cancel if price moved beyond tolerance_frac.
        """
        lower_bound = target_price * (Decimal('1') - self.tolerance_frac)
        upper_bound = target_price * (Decimal('1') + self.tolerance_frac)
        
        return order.price < lower_bound or order.price > upper_bound
    
    async def make_decision(self, side: str, orderbook: dict) -> Decision:
        """Make trading decision for one side."""
        # Extract market data
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return Decision('nothing', ['no_market_data'])
        
        best_bid = Decimal(str(orderbook['bids'][0][0]))
        best_ask = Decimal(str(orderbook['asks'][0][0]))
        mid_price = (best_bid + best_ask) / 2
        
        # Update volatility estimate
        if self.state.last_mid_price:
            self.update_volatility(mid_price)
        self.state.last_mid_price = mid_price
        
        # Check lockouts
        lockout_reasons = self.check_lockouts(side)
        if lockout_reasons:
            # If locked out, cancel any open order
            open_order = self.state.buy_order if side == 'buy' else self.state.sell_order
            if open_order and not open_order.cancel_pending:
                return Decision('cancel', lockout_reasons, order_id=open_order.order_id)
            return Decision('nothing', lockout_reasons)
        
        # Calculate AS optimal prices
        reservation_price = self.calculate_reservation_price(mid_price)
        optimal_spread = self.calculate_optimal_spread()
        
        if side == 'buy':
            target_price = reservation_price - optimal_spread / 2
            # Don't cross the spread
            target_price = min(target_price, best_bid)
        else:
            target_price = reservation_price + optimal_spread / 2
            # Don't cross the spread
            target_price = max(target_price, best_ask)
        
        # Round to tick size (simplified - should use market's actual tick size)
        tick_size = Decimal('0.01')
        target_price = (target_price / tick_size).quantize(Decimal('1')) * tick_size
        
        # Check existing order
        open_order = self.state.buy_order if side == 'buy' else self.state.sell_order
        
        if open_order:
            if open_order.cancel_pending:
                return Decision('nothing', ['cancel_pending'])
            
            # Check if order needs update based on tolerance
            if self.should_cancel_order(open_order, target_price):
                return Decision('cancel', ['outside_tolerance'], order_id=open_order.order_id)
            else:
                return Decision('nothing', ['within_tolerance'])
        
        # Check position limits
        position_reasons = self.check_position_limits(side, self.order_size)
        if position_reasons:
            return Decision('nothing', position_reasons)
        
        # Send new order
        return Decision('send', ['optimal_price'], price=target_price, quantity=self.order_size)
    
    def update_volatility(self, new_price: Decimal):
        """Update volatility estimate using EWMA."""
        if self.state.last_mid_price and self.state.last_mid_price > 0:
            log_return = abs(float(math.log(float(new_price / self.state.last_mid_price))))
            
            # EWMA with lambda = 0.94
            lambda_param = 0.94
            self.state.volatility = Decimal(str(
                lambda_param * float(self.state.volatility) + 
                (1 - lambda_param) * log_return * 100  # Annualize roughly
            ))
            
            # Clamp volatility to reasonable range
            self.state.volatility = max(Decimal('0.001'), min(Decimal('0.5'), self.state.volatility))
    
    async def on_orderbook_update(self, orderbook: dict):
        """Handle orderbook update - main strategy logic."""
        async with self.state_lock:
            # Make decisions for both sides
            buy_decision = await self.make_decision('buy', orderbook)
            sell_decision = await self.make_decision('sell', orderbook)
            
            # Log decisions with reasons (from Rust pattern)
            logger.debug(f"Buy: {buy_decision.action} ({', '.join(buy_decision.reasons)})")
            logger.debug(f"Sell: {sell_decision.action} ({', '.join(sell_decision.reasons)})")
            
            # Return decisions for execution
            return [buy_decision, sell_decision]
    
    async def on_fill(self, order_id: str, side: str, price: Decimal, quantity: Decimal):
        """Handle order fill."""
        await self.update_state(lambda s: self._handle_fill(s, order_id, side, price, quantity))
    
    def _handle_fill(self, state: MarketState, order_id: str, side: str, 
                     price: Decimal, quantity: Decimal):
        """Process fill (internal)."""
        # Update position
        if side == 'buy':
            state.position += quantity
            state.buy_order = None
        else:
            state.position -= quantity
            state.sell_order = None
        
        # Update lockout
        state.last_fill_time[side] = datetime.now()
        
        # Update metrics
        state.total_volume += quantity
        state.fill_count += 1
        
        # Simple P&L tracking
        if side == 'sell':
            state.realized_pnl += price * quantity
        else:
            state.realized_pnl -= price * quantity
        
        logger.info(f"Fill: {side} {quantity} @ {price} | Position: {state.position} | PnL: {state.realized_pnl}")
    
    async def on_order_rejected(self, order_id: str, side: str, reason: str):
        """Handle order rejection."""
        await self.update_state(lambda s: self._handle_reject(s, order_id, side))
    
    def _handle_reject(self, state: MarketState, order_id: str, side: str):
        """Process rejection (internal)."""
        # Clear order
        if side == 'buy':
            state.buy_order = None
        else:
            state.sell_order = None
        
        # Update lockout
        state.last_reject_time[side] = datetime.now()
        state.reject_count += 1
        
        logger.warning(f"Order rejected: {side} order {order_id}")
    
    async def on_order_cancelled(self, order_id: str, side: str):
        """Handle order cancellation confirmation."""
        await self.update_state(lambda s: self._handle_cancel(s, order_id, side))
    
    def _handle_cancel(self, state: MarketState, order_id: str, side: str):
        """Process cancellation (internal)."""
        if side == 'buy':
            state.buy_order = None
        else:
            state.sell_order = None
        logger.debug(f"Order cancelled: {side} order {order_id}")
    
    async def on_order_sent(self, order_id: str, side: str, price: Decimal, quantity: Decimal):
        """Track order that was sent."""
        await self.update_state(lambda s: self._handle_order_sent(s, order_id, side, price, quantity))
    
    def _handle_order_sent(self, state: MarketState, order_id: str, side: str, 
                           price: Decimal, quantity: Decimal):
        """Track sent order (internal)."""
        order = OpenOrder(order_id, side, price, quantity, datetime.now())
        
        if side == 'buy':
            state.buy_order = order
        else:
            state.sell_order = order
        
        state.last_order_time[side] = datetime.now()
        state.order_count += 1
        
        logger.info(f"Order sent: {side} {quantity} @ {price}")
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        return {
            'position': float(self.state.position),
            'realized_pnl': float(self.state.realized_pnl),
            'total_volume': float(self.state.total_volume),
            'volatility': float(self.state.volatility),
            'fill_count': self.state.fill_count,
            'order_count': self.state.order_count,
            'reject_count': self.state.reject_count,
            'fill_rate': self.state.fill_count / max(1, self.state.order_count),
        }
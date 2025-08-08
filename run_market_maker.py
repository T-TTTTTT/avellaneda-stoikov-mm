#!/usr/bin/env python3
"""
Run the Avellaneda-Stoikov Market Maker
Connects to Lighter CPTY and executes the strategy.
"""
import asyncio
import logging
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Add parent directory to path for imports
sys.path.append('/home/ec2-user/lighter-cpty')

from avellaneda_stoikov import AvellanedaStoikovMM
from LighterCpty.lighter_cpty_async import LighterCpty
from architect_py import Order, OrderDir, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketMakerRunner:
    """Connects AS strategy to Lighter CPTY."""
    
    def __init__(self, config_path: str):
        """Initialize with config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.strategies = {}
        self.cpty = None
        self.running = False
        
        # Track orders for each market
        self.pending_orders = {}  # order_id -> (market, side)
        
    async def initialize(self):
        """Initialize CPTY connection and strategies."""
        logger.info("Initializing market maker...")
        
        # Initialize CPTY
        self.cpty = LighterCpty()
        await self.cpty.login(
            user_id="trader1",  # Should be in config
            account_id="30188"  # Should be in config
        )
        
        # Initialize strategies for each enabled market
        for market_name, market_config in self.config['markets'].items():
            if market_config.get('enabled', False):
                logger.info(f"Initializing strategy for {market_name}")
                self.strategies[market_name] = AvellanedaStoikovMM(market_config)
        
        # Set up WebSocket callbacks
        self.cpty.ws_client.on_order_book = self._on_orderbook_update
        
        # Subscribe to orderbooks
        for market_name in self.strategies.keys():
            market_id = self._get_market_id(market_name)
            if market_id is not None:
                await self.cpty.ws_client.subscribe_order_book(market_id)
                logger.info(f"Subscribed to orderbook for {market_name} (ID: {market_id})")
    
    def _get_market_id(self, market_name: str) -> int:
        """Map market name to ID (simplified - should use proper mapping)."""
        market_map = {
            'BTC-USDC': 0,
            'ETH-USDC': 1,
            'HYPE-USDC': 24,
            'SOL-USDC': 2,
        }
        return market_map.get(market_name)
    
    def _get_market_name(self, market_id: int) -> str:
        """Map market ID to name."""
        id_map = {
            0: 'BTC-USDC',
            1: 'ETH-USDC',
            24: 'HYPE-USDC',
            2: 'SOL-USDC',
        }
        return id_map.get(market_id, f'UNKNOWN-{market_id}')
    
    async def _on_orderbook_update(self, market_id: int, orderbook: dict):
        """Handle orderbook updates from WebSocket."""
        market_name = self._get_market_name(market_id)
        
        if market_name not in self.strategies:
            return
        
        strategy = self.strategies[market_name]
        
        try:
            # Get decisions from strategy
            decisions = await strategy.on_orderbook_update(orderbook)
            
            # Execute decisions
            for decision in decisions:
                await self._execute_decision(market_name, decision)
                
        except Exception as e:
            logger.error(f"Error processing orderbook for {market_name}: {e}")
    
    async def _execute_decision(self, market_name: str, decision):
        """Execute a trading decision."""
        if decision.action == 'send':
            await self._send_order(
                market_name,
                'buy' if 'buy' in str(decision.reasons) else 'sell',
                decision.price,
                decision.quantity
            )
        elif decision.action == 'cancel':
            await self._cancel_order(decision.order_id)
    
    async def _send_order(self, market_name: str, side: str, price: Decimal, quantity: Decimal):
        """Send an order to the exchange."""
        try:
            # Generate order ID
            order_id = f"AS_{market_name}_{side}_{datetime.now().strftime('%H%M%S%f')}"
            
            # Create order
            order = Order(
                cl_ord_id=order_id,
                symbol=f"{market_name} LIGHTER Perpetual/USDC Crypto",
                dir=OrderDir.BUY if side == 'buy' else OrderDir.SELL,
                price=str(price),
                qty=str(quantity),
                type=OrderType.LIMIT,
                post_only=True  # Maker only
            )
            
            # Track order
            self.pending_orders[order_id] = (market_name, side)
            
            # Send via CPTY
            result = await self.cpty.place_order(order)
            
            # Notify strategy
            strategy = self.strategies[market_name]
            await strategy.on_order_sent(order_id, side, price, quantity)
            
            logger.info(f"Sent {side} order for {market_name}: {quantity} @ {price}")
            
        except Exception as e:
            logger.error(f"Failed to send order: {e}")
    
    async def _cancel_order(self, order_id: str):
        """Cancel an order."""
        try:
            await self.cpty.cancel_order(order_id)
            logger.info(f"Cancelled order {order_id}")
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
    
    async def _handle_fill(self, fill):
        """Handle order fill from CPTY."""
        order_id = fill.cl_ord_id
        if order_id in self.pending_orders:
            market_name, side = self.pending_orders[order_id]
            strategy = self.strategies[market_name]
            
            await strategy.on_fill(
                order_id,
                side,
                Decimal(str(fill.price)),
                Decimal(str(fill.qty))
            )
            
            # Remove from pending
            del self.pending_orders[order_id]
    
    async def _handle_reject(self, reject):
        """Handle order rejection from CPTY."""
        order_id = reject.cl_ord_id
        if order_id in self.pending_orders:
            market_name, side = self.pending_orders[order_id]
            strategy = self.strategies[market_name]
            
            await strategy.on_order_rejected(order_id, side, reject.reason)
            
            # Remove from pending
            del self.pending_orders[order_id]
    
    async def run_metrics_loop(self):
        """Periodically log metrics."""
        interval = self.config.get('monitoring', {}).get('metrics_interval_seconds', 60)
        
        while self.running:
            await asyncio.sleep(interval)
            
            # Collect metrics from all strategies
            all_metrics = {}
            for market_name, strategy in self.strategies.items():
                all_metrics[market_name] = strategy.get_metrics()
            
            # Log metrics
            logger.info("=== Performance Metrics ===")
            total_pnl = Decimal('0')
            for market, metrics in all_metrics.items():
                logger.info(f"{market}: Position={metrics['position']:.4f}, "
                          f"PnL=${metrics['realized_pnl']:.2f}, "
                          f"Fill Rate={metrics['fill_rate']:.1%}")
                total_pnl += Decimal(str(metrics['realized_pnl']))
            logger.info(f"Total PnL: ${total_pnl:.2f}")
            
            # Save to file if configured
            if self.config.get('monitoring', {}).get('save_metrics_to_file', False):
                with open('metrics.json', 'w') as f:
                    json.dump(all_metrics, f, indent=2, default=str)
    
    async def run(self):
        """Main run loop."""
        self.running = True
        
        try:
            # Initialize
            await self.initialize()
            
            # Start metrics loop
            metrics_task = asyncio.create_task(self.run_metrics_loop())
            
            logger.info("Market maker running. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.running = False
        finally:
            # Clean shutdown
            if self.cpty:
                # Cancel all open orders
                for strategy in self.strategies.values():
                    if strategy.state.buy_order:
                        await self._cancel_order(strategy.state.buy_order.order_id)
                    if strategy.state.sell_order:
                        await self._cancel_order(strategy.state.sell_order.order_id)
                
                # Disconnect
                await self.cpty.logout()
            
            logger.info("Market maker stopped.")


async def main():
    """Entry point."""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    runner = MarketMakerRunner(str(config_path))
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Run the Avellaneda-Stoikov Market Maker using Architect Client
Connects to Architect Core which routes orders to appropriate exchanges.
"""
import asyncio
import logging
import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from architect_py.async_client import AsyncClient
from architect_py import Order, OrderDir, OrderType
from avellaneda_stoikov import AvellanedaStoikovMM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketMakerRunner:
    """Runs AS strategy using Architect Client."""
    
    def __init__(self, config_path: str):
        """Initialize with config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.strategies = {}
        self.client: Optional[AsyncClient] = None
        self.running = False
        
        # Track orders for each market
        self.pending_orders = {}  # order_id -> (market, side)
        
        # Subscription tasks
        self.subscription_tasks = []
        
    async def initialize(self):
        """Initialize Architect connection and strategies."""
        logger.info("Initializing market maker...")
        
        # Get Architect credentials from environment or config
        architect_config = self.config.get('architect', {})
        
        # Connect to Architect Core (not CPTY!)
        logger.info("Connecting to Architect Core...")
        self.client = await AsyncClient.connect(
            endpoint=architect_config.get('endpoint', os.getenv('ARCHITECT_HOST')),
            api_key=architect_config.get('api_key', os.getenv('ARCHITECT_API_KEY')),
            api_secret=architect_config.get('api_secret', os.getenv('ARCHITECT_API_SECRET')),
            paper_trading=architect_config.get('paper_trading', False),
            use_tls=architect_config.get('use_tls', True),
        )
        logger.info("✓ Connected to Architect Core")
        
        # Initialize strategies for each enabled market
        for market_name, market_config in self.config['markets'].items():
            if market_config.get('enabled', False):
                logger.info(f"Initializing strategy for {market_name}")
                self.strategies[market_name] = AvellanedaStoikovMM(market_config)
        
        # Subscribe to market data for each strategy
        await self._subscribe_to_markets()
    
    async def _subscribe_to_markets(self):
        """Subscribe to market data via Architect."""
        for market_name, market_config in self.config['markets'].items():
            if market_config.get('enabled', False):
                venue = market_config.get('venue', 'LIGHTER')
                symbol = self._get_symbol_for_architect(market_name, venue)
                
                logger.info(f"Subscribing to {symbol} on {venue}")
                
                # Create subscription task
                task = asyncio.create_task(
                    self._market_data_loop(market_name, symbol, venue)
                )
                self.subscription_tasks.append(task)
    
    def _get_symbol_for_architect(self, market_name: str, venue: str) -> str:
        """Get properly formatted symbol for Architect."""
        # For Architect, we need the full symbol format
        # Format: "BASE-QUOTE VENUE Type/Settlement Currency"
        base, quote = market_name.split('-')
        
        # Venue-specific formatting
        if venue == 'LIGHTER':
            return f"{base}-{quote} LIGHTER Perpetual/{quote} Crypto"
        elif venue == 'BINANCE':
            return f"{base}-{quote} BINANCE Spot/{quote} Crypto"
        else:
            # Default format
            return f"{base}-{quote} {venue} Spot/{quote} Crypto"
    
    async def _market_data_loop(self, market_name: str, symbol: str, venue: str):
        """Subscribe to and process market data for a single market."""
        strategy = self.strategies[market_name]
        
        try:
            while self.running:
                try:
                    # Get L1 book snapshot (best bid/ask)
                    # In production, you'd want streaming updates
                    snapshot = await self.client.get_l1_book_snapshot(
                        symbol=symbol,
                        venue=venue
                    )
                    
                    if snapshot and snapshot.best_bid and snapshot.best_ask:
                        # Convert to orderbook format expected by strategy
                        orderbook = {
                            'bids': [[snapshot.best_bid.price, snapshot.best_bid.quantity]],
                            'asks': [[snapshot.best_ask.price, snapshot.best_ask.quantity]]
                        }
                        
                        # Get decisions from strategy
                        decisions = await strategy.on_orderbook_update(orderbook)
                        
                        # Execute decisions
                        for decision in decisions:
                            await self._execute_decision(market_name, symbol, venue, decision)
                    
                    # Poll interval (in production, use streaming)
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in market data loop for {market_name}: {e}")
                    await asyncio.sleep(5)  # Back off on error
                    
        except asyncio.CancelledError:
            logger.info(f"Market data loop cancelled for {market_name}")
    
    async def _execute_decision(self, market_name: str, symbol: str, venue: str, decision):
        """Execute a trading decision via Architect."""
        if decision.action == 'send':
            side = 'buy' if any('buy' in str(r).lower() for r in decision.reasons) else 'sell'
            await self._send_order(
                market_name,
                symbol,
                venue,
                side,
                decision.price,
                decision.quantity
            )
        elif decision.action == 'cancel':
            await self._cancel_order(decision.order_id)
    
    async def _send_order(self, market_name: str, symbol: str, venue: str, 
                          side: str, price: Decimal, quantity: Decimal):
        """Send an order via Architect Client."""
        try:
            # Generate order ID
            order_id = f"AS_{market_name}_{side}_{datetime.now().strftime('%H%M%S%f')}"
            
            # Create order using Architect's Order class
            order = Order(
                cl_ord_id=order_id,
                symbol=symbol,
                venue=venue,  # Architect routes to correct exchange
                dir=OrderDir.BUY if side == 'buy' else OrderDir.SELL,
                price=str(price),
                qty=str(quantity),
                type=OrderType.LIMIT,
                post_only=True  # Maker only
            )
            
            # Track order
            self.pending_orders[order_id] = (market_name, side)
            
            # Send via Architect Client
            result = await self.client.place_order(order)
            
            # Notify strategy
            strategy = self.strategies[market_name]
            await strategy.on_order_sent(order_id, side, price, quantity)
            
            logger.info(f"Sent {side} order for {market_name}: {quantity} @ {price} via {venue}")
            
        except Exception as e:
            logger.error(f"Failed to send order: {e}")
    
    async def _cancel_order(self, order_id: str):
        """Cancel an order via Architect Client."""
        try:
            await self.client.cancel_order(order_id)
            logger.info(f"Cancelled order {order_id}")
            
            # Notify strategy if we know which one
            if order_id in self.pending_orders:
                market_name, side = self.pending_orders[order_id]
                strategy = self.strategies[market_name]
                await strategy.on_order_cancelled(order_id, side)
                del self.pending_orders[order_id]
                
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
    
    async def _monitor_orders(self):
        """Monitor order status via Architect."""
        while self.running:
            try:
                # Get order updates from Architect
                # This would ideally be a streaming subscription
                orders = await self.client.get_open_orders()
                
                for order in orders:
                    # Process order updates
                    if order.cl_ord_id in self.pending_orders:
                        market_name, side = self.pending_orders[order.cl_ord_id]
                        strategy = self.strategies[market_name]
                        
                        # Check for fills
                        if order.status == 'FILLED':
                            await strategy.on_fill(
                                order.cl_ord_id,
                                side,
                                Decimal(str(order.avg_fill_price)),
                                Decimal(str(order.filled_qty))
                            )
                            del self.pending_orders[order.cl_ord_id]
                        
                        # Check for rejections
                        elif order.status == 'REJECTED':
                            await strategy.on_order_rejected(
                                order.cl_ord_id,
                                side,
                                order.reject_reason or 'Unknown'
                            )
                            del self.pending_orders[order.cl_ord_id]
                
                await asyncio.sleep(1)  # Check interval
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(5)
    
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
            
            # Start background tasks
            monitor_task = asyncio.create_task(self._monitor_orders())
            metrics_task = asyncio.create_task(self.run_metrics_loop())
            
            logger.info("Market maker running. Press Ctrl+C to stop.")
            
            # Wait for all tasks
            await asyncio.gather(
                monitor_task,
                metrics_task,
                *self.subscription_tasks
            )
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.running = False
        finally:
            # Clean shutdown
            self.running = False
            
            # Cancel all tasks
            for task in self.subscription_tasks:
                task.cancel()
            
            if self.client:
                # Cancel all open orders
                for market_name, strategy in self.strategies.items():
                    if strategy.state.buy_order:
                        await self._cancel_order(strategy.state.buy_order.order_id)
                    if strategy.state.sell_order:
                        await self._cancel_order(strategy.state.sell_order.order_id)
                
                # Disconnect from Architect
                await self.client.close()
            
            logger.info("Market maker stopped.")


async def main():
    """Entry point."""
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Load environment variables if .env exists
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    runner = MarketMakerRunner(str(config_path))
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
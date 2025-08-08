# Avellaneda-Stoikov Market Maker

A lean, production-ready implementation of the Avellaneda-Stoikov optimal market making strategy for the Lighter exchange.

## Overview

This market maker combines:
- **Mathematical optimality** from the Avellaneda-Stoikov (2008) model
- **Practical engineering patterns** from production Rust market makers
- **Clean async Python** architecture for simplicity and maintainability

## Mathematical Model

The Avellaneda-Stoikov model calculates optimal bid and ask prices to maximize expected utility while managing inventory risk:

### Reservation Price
```
r(t) = s(t) - q(t) * γ * σ² * (T - t)
```
- `s(t)`: Mid-market price
- `q(t)`: Current inventory position
- `γ`: Risk aversion parameter (0.01 to 1.0)
- `σ`: Volatility estimate
- `T - t`: Time remaining in trading session

### Optimal Spread
```
δ* = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
```
- `k`: Order arrival rate parameter

### Final Quotes
```
bid_price = r(t) - δ*/2
ask_price = r(t) + δ*/2
```

## Features

### From Avellaneda-Stoikov Theory
- Dynamic spread calculation based on volatility and time
- Inventory-based price skewing
- Risk-averse position management
- EWMA volatility estimation

### From Rust MM Best Practices
- **Decision tracking**: Every action has traceable reasons
- **Lockout mechanisms**: Prevents overtrading after fills/rejects
- **Tolerance bands**: Avoids excessive order updates
- **State management**: All updates through controlled mutations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd avellaneda-stoikov-mm

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

Edit `config.yaml` to configure markets and parameters:

```yaml
markets:
  BTC-USDC:
    enabled: true
    
    # Avellaneda-Stoikov parameters
    gamma: 0.1          # Risk aversion (0.01 to 1.0)
    volatility: 0.015   # Initial volatility estimate
    
    # Position limits
    max_position: 0.1   # Maximum position size
    order_size: 0.001   # Size per order
    
    # Lockouts (from Rust pattern)
    fill_lockout_ms: 1000    # Wait after fills
    order_lockout_ms: 500    # Wait between orders
    reject_lockout_ms: 2000  # Wait after rejects
    
    # Tolerance band
    tolerance_frac: 0.001    # Cancel if price moves > 0.1%
```

## Usage

### Prerequisites

1. Lighter CPTY must be running at `localhost:50051`
   ```bash
   # In lighter-cpty directory
   tmux new -s l_c
   python run_lighter_cpty.py
   ```

2. Ensure you have valid credentials configured

### Running the Market Maker

```bash
# Run with uv
uv run python run_market_maker.py

# Or directly if installed
python run_market_maker.py
```

## Architecture

### Components

1. **`avellaneda_stoikov.py`** - Core strategy implementation
   - AS pricing model
   - Decision engine with reasons
   - State management
   - Risk controls

2. **`run_market_maker.py`** - Main execution loop
   - Connects to Lighter CPTY
   - Handles orderbook updates
   - Executes trading decisions
   - Metrics logging

3. **`utils.py`** - Helper functions
   - Price/quantity rounding
   - Volatility estimation
   - Performance metrics

### Decision Flow

```
Orderbook Update
    ↓
Calculate AS Prices
    ↓
Check Lockouts & Limits
    ↓
Make Decision (with reasons)
    ↓
Send/Cancel Orders
    ↓
Update State
```

### Example Decision Output

```
Buy: send (optimal_price)
Sell: nothing (fill_lockout, max_position)
```

## Metrics

The market maker tracks and logs:
- Position and P&L
- Fill rate
- Volatility estimates
- Order counts and rejects
- Volume traded

Metrics are logged every 60 seconds and optionally saved to `metrics.json`.

## Risk Management

- **Position limits**: Configurable min/max positions
- **Lockout periods**: Prevents overtrading
- **Tolerance bands**: Reduces order updates
- **Daily loss limits**: Optional stop-loss (configure in config.yaml)

## Development

### Testing

```bash
# Run tests (when implemented)
uv run pytest tests/
```

### Code Quality

```bash
# Format code
uv run black .

# Lint
uv run ruff check .

# Type checking
uv run mypy .
```

## Performance

- **Single process**: Low latency, no IPC overhead
- **Async I/O**: Non-blocking orderbook processing
- **Efficient state updates**: Locked, atomic mutations
- **Minimal dependencies**: Lean and fast

## References

- Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book"
- Original Rust MM implementation patterns from arch_rust_src

## License

Proprietary - See LICENSE file

## Support

For issues or questions, please open an issue in the repository.
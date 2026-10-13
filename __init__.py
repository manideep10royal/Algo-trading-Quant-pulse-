"""Portfolio optimization sub-package.

Self-contained quantitative finance modules:
- portfolio_optimizer : Modern Portfolio Theory + Max Sharpe / Min Variance
- efficient_frontier  : Monte Carlo random portfolios for the frontier
- monte_carlo         : Forward portfolio simulation
- risk_parity         : Equal risk-contribution allocation
- optimization_metrics: Sharpe / Sortino / Max Drawdown / Diversification ratio
- allocation_engine   : GPT-4o-powered AI portfolio analyst
- rebalance_engine    : Trade-list generation from drift
- correlation_analysis: Pairwise correlations & heatmap data
"""

from .portfolio_optimizer import PortfolioOptimizer, OptimizationResult
from .efficient_frontier import EfficientFrontier
from .monte_carlo import MonteCarloSimulator
from .risk_parity import RiskParity
from .optimization_metrics import OptimizationMetrics
from .allocation_engine import AIAllocationEngine
from .rebalance_engine import RebalanceEngine
from .correlation_analysis import CorrelationAnalysis

__all__ = [
    "PortfolioOptimizer", "OptimizationResult", "EfficientFrontier",
    "MonteCarloSimulator", "RiskParity", "OptimizationMetrics",
    "AIAllocationEngine", "RebalanceEngine", "CorrelationAnalysis",
]

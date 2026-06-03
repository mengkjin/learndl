# Strategy Development Workflow
**Purpose:** Checklist for developing and validating a new alpha signal end-to-end, from idea to live portfolio.

---

## Stage 1 — Factor Development

1. Create a new `FactorCalculator` subclass in `src/factor/`:
   ```python
   class MyNewFactor(MomentumBase):
       name = 'my_new_factor'
       params = {'window': 20}

       def calculate(self, data: DataBlock) -> DataBlock:
           ...
   ```
2. Test calculation on a short date range:
   ```python
   FactorAPI.calculate('my_new_factor', dates=recent_dates, universe='hs300')
   ```
3. Run factor tests:
   ```python
   FactorAPI.Test.run_all('my_new_factor', dates, universe)
   ```
   Check IC, ICIR, quantile spreads. Target: ICIR > 0.5 on out-of-sample data.

4. If IC looks good, run full history update:
   ```python
   FactorAPI.update('my_new_factor', start='20150101', end='today')
   ```

---

## Stage 2 — Model Training (optional)

If the factor will be used as a model feature (rather than directly as a signal):

1. Add the factor to a schedule config's `data.features` list
2. Train the model: `ModelAPI.train_model(schedule_name)`
3. Check model ICIR: `SummaryAPI.model_account_summary(schedule_name)`

---

## Stage 3 — Portfolio Backtest

1. Add a portfolio config to `configs/trading/trading_port.yaml`:
   ```yaml
   - name: top50_my_factor
     alpha: my_new_factor      # or a model schedule name
     universe: hs300
     category: top
     top_num: 50
     freq: daily
     benchmark: 000300.SH
     buffer_zone: 0.2
     indus_control: false
   ```
2. Run backtest:
   ```python
   TradingAPI.Backtest('top50_my_factor')
   ```
3. Analyze results:
   ```python
   TradingAPI.Analyze('top50_my_factor')
   SummaryAPI.backtest_port_account_summary('top50_my_factor')
   ```
   Check: annualized return, Sharpe ratio, max drawdown, turnover.

---

## Stage 4 — Live Tracking

Once backtest results are satisfactory:

1. Ensure the portfolio entry is in `configs/trading/trading_port.yaml`
2. Add to `TrackingPortfolioManager` list
3. Run daily via `TradingAPI.update()` (see [[daily_update]])

---

## Decision Criteria

| Metric | Threshold |
|--------|-----------|
| Factor ICIR (in-sample) | > 0.5 |
| Factor ICIR (out-of-sample) | > 0.3 |
| Backtest Sharpe (annualized) | > 1.0 |
| Max drawdown | < 30% |
| One-way turnover | < 50% per month |

These are guidelines, not hard rules. Context matters — compare against existing strategies.

---

## See Also
- [[factor_engine]] — factor class API and normalization
- [[nn_models]] — model training details
- [[trading]] — portfolio construction and analysis

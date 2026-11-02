"""Pipeline for estimating grid battery revenue in WA's SWIS since the WEM reform.

Modules
-------
paths            repository directory layout and helpers
download_data    fetch AEMO public data files into data/raw/
clean            parse and validate raw AEMO CSVs into tidy parquet in data/processed/
model            co-optimised energy + FCESS battery dispatch LP, solved with HiGHS
strategies       perfect foresight and rolling-horizon dispatch strategies
backcast         rebuild what the real SWIS batteries actually earned
price_impact     spread evolution and residual-supply price-taker analysis
economics        NPV, IRR, degradation and sensitivity analysis
"""

__all__ = [
    "paths",
]

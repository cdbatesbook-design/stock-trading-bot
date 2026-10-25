def research(
    data_dir: Path,
    output_dir: Path,
    config: Config,
    symbols=None,
    period="1y",
    interval="1h",
):
    """
    Download historical market data automatically.
    CSV files are not required.

    Research only. No live orders are placed.
    """

    if symbols is None:
        symbols = [
            "AAPL",
            "MSFT",
            "NVDA",
            "SPY",
            "QQQ",
            "AMD",
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
        ]

    data = {}

    print("\n======================================")
    print("MICRO-PROFIT MARKET RESEARCH")
    print("======================================")
    print("Downloading historical market data...")
    print(f"Period: {period}")
    print(f"Interval: {interval}")
    print(f"Assets requested: {len(symbols)}")

    for symbol in symbols:
        print(f"\nDownloading {symbol}...")

        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if raw is None or raw.empty:
                print(f"SKIPPING {symbol}: no data returned")
                continue

            # Handle different yfinance column formats.
            if isinstance(raw.columns, pd.MultiIndex):
                flattened = {}

                for col in raw.columns:
                    parts = [
                        str(part)
                        for part in col
                    ]

                    price_column = next(
                        (
                            part
                            for part in parts
                            if part.lower() in {
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            }
                        ),
                        None,
                    )

                    if price_column is not None:
                        flattened[col] = price_column

                raw.columns = [
                    flattened.get(col, str(col))
                    for col in raw.columns
                ]

            # Remove duplicate column names if present.
            raw = raw.loc[
                :,
                ~raw.columns.duplicated()
            ].copy()

            raw = raw.reset_index()

            # yfinance can name the timestamp column
            # either Date or Datetime.
            time_column = next(
                (
                    col
                    for col in raw.columns
                    if str(col).lower()
                    in {"date", "datetime", "timestamp"}
                ),
                None,
            )

            if time_column is None:
                print(
                    f"SKIPPING {symbol}: "
                    "timestamp column missing"
                )
                continue

            raw = raw.rename(
                columns={
                    time_column: "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            required = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            if not all(
                col in raw.columns
                for col in required
            ):
                print(
                    f"SKIPPING {symbol}: "
                    "required price columns missing"
                )
                continue

            raw["timestamp"] = pd.to_datetime(
                raw["timestamp"],
                utc=True,
                errors="coerce",
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                raw[col] = pd.to_numeric(
                    raw[col],
                    errors="coerce",
                )

            raw = raw.dropna(
                subset=required
            )

            raw = raw.sort_values(
                "timestamp"
            )

            raw = raw.drop_duplicates(
                "timestamp"
            )

            raw = raw.reset_index(
                drop=True
            )

            if len(raw) < 100:
                print(
                    f"SKIPPING {symbol}: "
                    f"only {len(raw)} valid bars"
                )
                continue

            raw = add_indicators(
                raw,
                config,
            )

            data[symbol] = raw

            print(
                f"Loaded {len(raw)} bars for {symbol}"
            )

        except Exception as exc:
            print(
                f"SKIPPING {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not data:
        raise RuntimeError(
            "No usable market data was downloaded. "
            "Check internet access, yfinance installation, "
            "market-data availability, and GitHub Actions logs."
        )

    print("\n======================================")
    print("DATA DOWNLOAD COMPLETE")
    print("======================================")
    print(f"Successfully loaded: {len(data)} assets")

    train, validation, test = split_data(data)

    candidates = []

    strategies = [
        "momentum",
        "breakout",
        "mean_reversion",
    ]

    # TRAINING
    print("\nTesting candidate strategies...")

    for strategy in strategies:
        for target in config.profit_targets:
            result = run_backtest(
                train,
                strategy,
                target,
                config,
            )

            candidates.append(result)

    candidates.sort(
        key=lambda x: x["net_profit"],
        reverse=True,
    )

    # VALIDATION
    validation_results = []

    for candidate in candidates:
        result = run_backtest(
            validation,
            candidate["strategy"],
            candidate["target_fraction"],
            config,
        )

        validation_results.append(result)

    validation_results.sort(
        key=lambda x: x["net_profit"],
        reverse=True,
    )

    # Only candidates that meet the validation criteria
    # advance to the final test.
    finalists = [
        result
        for result in validation_results
        if (
            result["trade_count"] >= config.minimum_trades
            and result["net_profit"] > 0
            and (result["profit_factor"] or 0) > 1.0
            and result["max_drawdown"] > -0.20
        )
    ]

    # FINAL TEST
    final_results = []

    for candidate in finalists:
        result = run_backtest(
            test,
            candidate["strategy"],
            candidate["target_fraction"],
            config,
        )

        final_results.append(result)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def clean_result(result):
        return {
            key: value
            for key, value in result.items()
            if key not in (
                "trades",
                "equity_curve",
            )
        }

    report = {
        "mode": "RESEARCH_ONLY",
        "orders_enabled": False,
        "data_source": "yfinance",
        "period": period,
        "interval": interval,
        "config": asdict(config),
        "symbols": list(data.keys()),
        "bars": {
            symbol: len(df)
            for symbol, df in data.items()
        },
        "train_results": [
            clean_result(result)
            for result in candidates
        ],
        "validation_results": [
            clean_result(result)
            for result in validation_results
        ],
        "final_test_results": [
            clean_result(result)
            for result in final_results
        ],
        "warning": (
            "Historical results do not guarantee future "
            "performance. Fees, slippage, data limitations, "
            "and actual execution can materially change results."
        ),
    }

    report_path = (
        output_dir / "research_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    pd.DataFrame(
        report["train_results"]
    ).to_csv(
        output_dir / "train_results.csv",
        index=False,
    )

    pd.DataFrame(
        report["validation_results"]
    ).to_csv(
        output_dir / "validation_results.csv",
        index=False,
    )

    pd.DataFrame(
        report["final_test_results"]
    ).to_csv(
        output_dir / "final_test_results.csv",
        index=False,
    )

    print("\n======================================")
    print("RESEARCH COMPLETE")
    print("======================================")
    print(f"Assets loaded: {len(data)}")
    print(f"Report saved to: {report_path}")
    print("Live trading: DISABLED")

    if final_results:
        print("\nFINAL TEST RESULTS")

        for result in sorted(
            final_results,
            key=lambda x: x["net_profit"],
            reverse=True,
        ):
            print(
                f"{result['strategy']:16} "
                f"target={result['target_fraction']:.3%} "
                f"return={result['return_fraction']:.2%} "
                f"trades={result['trade_count']} "
                f"win_rate={result['win_rate']:.1%} "
                f"profit_factor={result['profit_factor']} "
                f"drawdown={result['max_drawdown']:.2%}"
            )
    else:
        print(
            "\nNo strategy passed validation. "
            "No strategy will be promoted."
        )

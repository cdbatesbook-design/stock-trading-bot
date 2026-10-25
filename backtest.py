def optimize_strategy(train, validation):
    """
    Diagnostic optimizer.

    Evaluates all candidates, saves every training result,
    displays the top 10 candidates, and evaluates the best
    candidates on validation data.

    A candidate is never considered proven profitable merely
    because it ranks first.
    """

    candidates = parameter_grid()

    log.info("=" * 65)
    log.info("DIAGNOSTIC STRATEGY OPTIMIZATION")
    log.info("Candidates to evaluate: %s", len(candidates))
    log.info("=" * 65)

    training_results = []

    # --------------------------------------------------------
    # 1. EVALUATE EVERY CANDIDATE ON TRAINING DATA
    # --------------------------------------------------------

    for index, params in enumerate(candidates, start=1):

        try:
            signals = add_signals(train, params)

            result = run_simulation(
                signals,
                params,
                starting_capital=5000,
            )

            trades = int(result["trades"])
            ret = safe_float(result["return_pct"])
            win_rate = safe_float(result["win_rate"])
            pf = safe_float(result["profit_factor"])
            drawdown = abs(
                safe_float(result["max_drawdown_pct"])
            )

            # Diagnostic ranking score.
            #
            # Unlike the previous scoring method, this does not
            # assign the same failing score to every strategy
            # that misses a trade-count or profitability threshold.
            #
            # This score is for ranking research candidates only.
            # It does not establish that a strategy is profitable.

            capped_pf = min(pf, 3.0)

            diagnostic_score = (
                ret
                - 0.75 * drawdown
                + 0.02 * capped_pf
            )

            training_results.append({
                "params": params,
                "diagnostic_score": diagnostic_score,
                "train_return": ret,
                "train_trades": trades,
                "train_win_rate": win_rate,
                "train_profit_factor": pf,
                "train_drawdown": drawdown,
            })

        except Exception as exc:
            log.exception(
                "Candidate %s failed: %s | Error: %s",
                index,
                params,
                exc,
            )

        if index % 5 == 0 or index == len(candidates):
            log.info(
                "Training progress: %s/%s",
                index,
                len(candidates),
            )

    # --------------------------------------------------------
    # 2. SAVE ALL TRAINING RESULTS
    # --------------------------------------------------------

    if not training_results:
        raise RuntimeError(
            "Every training candidate failed during execution. "
            "Check the error messages above."
        )

    training_results.sort(
        key=lambda item: (
            item["diagnostic_score"],
            item["train_trades"],
        ),
        reverse=True,
    )

    training_output = []

    for item in training_results:
        row = dict(item)
        row["params"] = json.dumps(row["params"])
        training_output.append(row)

    training_path = os.path.join(
        OUTPUT_DIR,
        "training_results.csv",
    )

    pd.DataFrame(training_output).to_csv(
        training_path,
        index=False,
    )

    log.info(
        "All training results saved to %s",
        training_path,
    )

    # --------------------------------------------------------
    # 3. DISPLAY THE TOP 10 CANDIDATES
    # --------------------------------------------------------

    log.info("=" * 100)
    log.info("TOP 10 TRAINING CANDIDATES — NOT YET APPROVED")
    log.info("=" * 100)

    log.info(
        "%-4s %-10s %-10s %-10s %-12s %-12s %-12s",
        "Rank",
        "Return",
        "Trades",
        "Win rate",
        "Profit factor",
        "Drawdown",
        "Score",
    )

    for rank, item in enumerate(
        training_results[:10],
        start=1,
    ):
        log.info(
            "%-4s %-10.2f%% %-10s %-10.2f%% %-12.3f %-12.2f%% %-12.5f",
            rank,
            100 * item["train_return"],
            item["train_trades"],
            100 * item["train_win_rate"],
            item["train_profit_factor"],
            100 * item["train_drawdown"],
            item["diagnostic_score"],
        )

        log.info(
            "     Parameters: %s",
            item["params"],
        )

    # --------------------------------------------------------
    # 4. CHECK WHETHER ANY CANDIDATES ARE WORTH VALIDATING
    # --------------------------------------------------------

    candidates_with_trades = [
        item
        for item in training_results
        if item["train_trades"] >= 5
    ]

    if not candidates_with_trades:
        raise RuntimeError(
            "NO CANDIDATE GENERATED AT LEAST 5 TRAINING TRADES. "
            "Inspect training_results.csv and the logs. "
            "The strategy signals or market conditions may be "
            "producing too few opportunities."
        )

    # Select the highest-ranked candidates for validation.
    # They do not need to be profitable in training to be
    # examined, but validation must independently pass.
    finalists = candidates_with_trades[
        :TOP_TRAIN_CANDIDATES
    ]

    log.info(
        "%s candidates will proceed to validation.",
        len(finalists),
    )

    # --------------------------------------------------------
    # 5. VALIDATE THE FINALISTS
    # --------------------------------------------------------

    validation_results = []

    for index, item in enumerate(finalists, start=1):

        params = item["params"]

        try:
            signals = add_signals(
                validation,
                params,
            )

            result = run_simulation(
                signals,
                params,
                starting_capital=5000,
            )

            passed = validation_passes(result)

            validation_row = {
                "params": params,
                "passed_validation": passed,
                "validation_return": safe_float(
                    result["return_pct"]
                ),
                "validation_trades": int(
                    result["trades"]
                ),
                "validation_win_rate": safe_float(
                    result["win_rate"]
                ),
                "validation_profit_factor": safe_float(
                    result["profit_factor"]
                ),
                "validation_drawdown": safe_float(
                    result["max_drawdown_pct"]
                ),
                "training_score": item["diagnostic_score"],
                "training_return": item["train_return"],
                "training_trades": item["train_trades"],
                "training_profit_factor": item[
                    "train_profit_factor"
                ],
            }

            validation_results.append(validation_row)

            log.info(
                "Validation %s/%s | Passed: %s | "
                "Return: %.2f%% | Trades: %s | "
                "Win rate: %.2f%% | PF: %.3f | "
                "Drawdown: %.2f%%",
                index,
                len(finalists),
                passed,
                100 * validation_row["validation_return"],
                validation_row["validation_trades"],
                100 * validation_row["validation_win_rate"],
                validation_row["validation_profit_factor"],
                100 * validation_row["validation_drawdown"],
            )

        except Exception as exc:
            log.exception(
                "Validation failed for parameters %s: %s",
                params,
                exc,
            )

    # --------------------------------------------------------
    # 6. SAVE VALIDATION RESULTS
    # --------------------------------------------------------

    validation_path = os.path.join(
        OUTPUT_DIR,
        "validation_results.csv",
    )

    if validation_results:
        validation_output = []

        for item in validation_results:
            row = dict(item)
            row["params"] = json.dumps(row["params"])
            validation_output.append(row)

        pd.DataFrame(validation_output).to_csv(
            validation_path,
            index=False,
        )

        log.info(
            "Validation results saved to %s",
            validation_path,
        )

    if not validation_results:
        raise RuntimeError(
            "No candidates completed validation. "
            "Inspect the execution errors above."
        )

    # --------------------------------------------------------
    # 7. DO NOT PROMOTE A LOSING VALIDATION STRATEGY
    # --------------------------------------------------------

    accepted = [
        item
        for item in validation_results
        if item["passed_validation"]
    ]

    log.info(
        "Candidates passing validation: %s of %s",
        len(accepted),
        len(validation_results),
    )

    if not accepted:
        log.error(
            "NO STRATEGY PASSED VALIDATION. "
            "The final test will NOT run. "
            "Review training_results.csv and "
            "validation_results.csv to identify why."
        )

        raise RuntimeError(
            "NO STRATEGY PASSED VALIDATION. "
            "No strategy will be promoted."
        )

    # Rank only candidates that passed every validation rule.
    accepted.sort(
        key=lambda item: (
            item["validation_return"]
            - 0.75 * abs(item["validation_drawdown"])
        ),
        reverse=True,
    )

    selected = accepted[0]

    log.info("=" * 65)
    log.info("VALIDATION APPROVED A CANDIDATE")
    log.info("Selected parameters: %s", selected["params"])
    log.info(
        "Validation return: %.2f%%",
        100 * selected["validation_return"],
    )
    log.info(
        "Validation trades: %s",
        selected["validation_trades"],
    )
    log.info(
        "Validation profit factor: %.3f",
        selected["validation_profit_factor"],
    )
    log.info(
        "Validation drawdown: %.2f%%",
        100 * selected["validation_drawdown"],
    )
    log.info(
        "This is a research result, not a guarantee of future profit."
    )
    log.info("=" * 65)

    return selected

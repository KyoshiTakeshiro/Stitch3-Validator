# Stitch3 Validator

Stitch3 Validator is a toolkit for Bitcast creators, built around two tools: **Tweet Validator**, which checks whether a draft X post is likely to satisfy a campaign brief before publishing, and **Engagement Value**, which shows what every other account's quote or retweet is actually worth to a specific creator on a given campaign. Instead of posting first and discovering afterward that a tweet failed validation — or guessing at who's actually worth engaging with — creators get instant, campaign-specific feedback for both.

**Why this project is useful**

Creators often lose engagement when a tweet fails brief validation. By the time they discover the issue, the original post has already gained traction, and a replacement tweet typically performs worse.

This tool reduces that risk by providing immediate feedback before a tweet is published, allowing creators to improve compliance while preserving the opportunity for maximum engagement.

**How this matches the real Bitcast validator**

This tool is built to replicate the actual validator's brief-evaluation logic as closely as possible, not just approximate it:

- **Same model and provider.** Evaluation runs on `Qwen/Qwen3-32B` via [Chutes](https://chutes.ai) — the same model and provider the production validator uses, not a different LLM standing in for it.
- **Same prompts.** All prompt versions the validator currently supports (v1, v2, v5) are transcribed directly from its source, and the correct version is selected automatically per brief (via each brief's `prompt_version` field), the same way the real validator does.
- **Deliberately stricter than the real validator's own multi-check strategy.** The real validator runs up to 3 independent evaluation checks per tweet and accepts it as compliant if *any* check passes. This tool runs the same 3 independent checks (including the same per-check text differentiator the validator uses internally, so each check is a genuinely independent judgment rather than 3 repeats of the same answer) but requires *all 3* to pass, stopping early at the first failure. This is an intentional deviation, not a fidelity gap: since a false "pass" here is far more costly to a creator than a false "fail" (they'd post believing it's compliant, only to have the real validator disagree), the tool is tuned to be a conservative predictor rather than an exact replica of the validator's own leniency.

Because of that, a tweet can occasionally fail here even though the real validator, using its own more lenient any-1-of-3 rule, would have passed it — that trade-off is intentional. Results can also vary slightly between runs on borderline tweets, since each of the 3 checks is an independent LLM judgment — that's expected, not a bug.

**Engagement Value**

The Engagement Value tab reproduces Bitcast's own public scoring formula so a creator can see, per campaign, what every other considered account's quote or retweet is actually worth to them. Pick an ecosystem (Bittensor / Perp DEXs / Prediction Markets) and a campaign, and the tool shows a ranked leaderboard of every account considered for that campaign's ecosystem snapshot. Enter your own X handle to get a personalized view: your influence, rank, and baseline score, plus a per-account quote/retweet value that accounts for prior-interaction history between you and each account (a higher "Ties" score between two accounts slightly discounts the value, discouraging accounts that already engage each other from inflating each other's numbers). This tool doesn't call an LLM at all — it's a direct, deterministic calculation against Bitcast's public campaign and ecosystem-map data, so no `CHUTES_API_KEY` is needed to use it.

**Getting Started**

1. **Clone this repository.**

   ```bash
   git clone https://github.com/KyoshiTakeshiro/Stitch3-Validator.git
   cd Stitch3-Validator
   ```

2. **Install the project dependencies.**

   Create a virtual environment and install the Python packages from `requirements.txt`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure your Chutes API key.**

   Evaluation runs on [Chutes](https://chutes.ai) (`Qwen/Qwen3-32B`) — the same model and provider the real Bitcast validator uses. Copy the example env file and fill in your key:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and set `CHUTES_API_KEY` to a real key from your [Chutes account](https://chutes.ai).

4. **Start the FastAPI backend.**

   ```bash
   uvicorn main:app --reload
   ```

   This serves both the API and the frontend on `http://localhost:8000`.

5. **Open the web interface in your browser.**

   Visit [http://localhost:8000](http://localhost:8000). You should see the Stitch3 Validator UI.

6. **Pick a tab, then use either tool.**

   The **Tweet Validator** and **Engagement Value** tabs each have their own ecosystem/campaign selector and remember your last selection independently — switch between them at any time.

   - **Tweet Validator**: choose a campaign from the **Campaign brief** dropdown, paste your draft tweet into the **Draft tweet** field, then click **Check against brief**. The tool runs up to 3 independent checks and only passes the tweet if *all 3* agree — stricter than the real validator's own any-1-of-3 rule (see above) — returning a pass/fail verdict, with a reason shown if it fails.
   - **Engagement Value**: choose a campaign, then enter your X handle to see your influence, rank, and baseline score for that campaign, plus a ranked table of what every other account's quote or retweet is worth to you. Optionally enter another account's handle to filter the table down to just them.

**Maintainers & Contributions**

This project is maintained by @KyoshiTakeshiro

Contributions are welcome. If you find a bug, have ideas for improvements, or want to add new functionality, feel free to open an issue or submit a pull request.

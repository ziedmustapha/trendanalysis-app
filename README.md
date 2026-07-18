# trendanalysis-app

## Secrets setup

API credentials and the Nestlé OpenAI gateway URL are **not** stored in source code.

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```
2. Fill in real values in `.env` (see also local `SECRETS_INVENTORY.txt` if you created it).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

`.env` is gitignored and must never be committed.

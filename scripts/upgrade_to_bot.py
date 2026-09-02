"""One-time upgrade of the Lichess account behind LICHESS_TOKEN to a BOT account."""
import os
import sys
from pathlib import Path

import berserk
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
token = os.environ.get("LICHESS_TOKEN")
if not token:
    sys.exit("LICHESS_TOKEN is not set")
client = berserk.Client(session=berserk.TokenSession(token))
acct = client.account.get()
if acct.get("title") == "BOT":
    print(f"{acct['username']} is already a BOT account")
    sys.exit(0)
client.account.upgrade_to_bot()
print(f"{acct['username']} upgraded to BOT")

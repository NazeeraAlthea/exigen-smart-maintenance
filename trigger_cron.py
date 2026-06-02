import urllib.request
import json
import sys

url = "http://localhost:3000/api/cron/rul"

print("Sending GET request to http://localhost:3000/api/cron/rul ...")
print("This will trigger the Next.js Cron job to process RUL updates.")
print("Check your Next.js console/terminal to monitor real-time progress!")
sys.stdout.flush()

try:
    # Set a long timeout (30 minutes) because of 43k+ active assets
    with urllib.request.urlopen(url, timeout=1800) as response:
        res = json.loads(response.read().decode('utf-8'))
        print("\nSuccess! Next.js Cron Response:")
        print(json.dumps(res, indent=4))
except Exception as e:
    print("\nError or Timeout triggering Next.js Cron Route:", e)

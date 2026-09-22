import requests
import ultra_user.utils.spiceLoader as spiceLoader
from datetime import datetime
from pathlib import Path


spiceLoad = spiceLoader.spiceLoader()
dateStart = datetime.fromisoformat("2025-11-10")
dateEnd = datetime.now()
tdif0 = dateStart - datetime.fromisoformat("2000-01-01T12:00:00")
tdif1 = dateEnd - datetime.fromisoformat("2000-01-01T12:00:00")
etStart = tdif0.total_seconds()
etEnd = tdif1.total_seconds()
response=spiceLoad.getFileList(etStart,etEnd,loud=True)



## the following taken from ChatGPT
url = "https://api.imap-mission.com/metakernel"

params = {
    "start_time": 816004800.0,
    "end_time": 843019057.55304,
    "list_files": True
}

response = requests.get(url, params=params, timeout=30)

# Raise an exception for HTTP errors (404, 500, etc.)
response.raise_for_status()


print("Request URL:")
print(response.url)

print("\nContent type:")
print(response.headers.get("Content-Type"))

# Assuming the API returns JSON
data = response.json()

print("\nResponse:")
print(data)


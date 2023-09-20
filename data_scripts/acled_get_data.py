import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ACLED_API_KEY")
email = os.getenv("EMAIL")

base_api_url = f"https://api.acleddata.com/acled/read?key={api_key}&email={email}"

query_filters = {
    "event_date": "2023-01-01|2023-08-29",
    "event_date_where": "BETWEEN",
    "country": "Ukraine",
    "event_type": "Explosions/Remote violence:OR:event_type=Violence against civilians:OR:event_type=Battles",
    "fields": "event_date|event_type|sub_event_type|actor1|actor2|admin1|admin2|admin3|location|latitude|longitude|notes|fatalities|civilian_targeting",
}

all_data = []

page_number = 1

while True:
    query_filters["page"] = page_number

    api_url = base_api_url + "".join([f"&{k}={v}" for k, v in query_filters.items()])

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()

        if len(data["data"]) == 0:
            break

        all_data.extend(data["data"])

        page_number += 1
    else:
        print(f"Request failed with status code {response.status_code}.")
        break

with open("acled_data.json", "w") as json_file:
    json.dump(all_data, json_file)

print("Data saved to 'acled_data.json'.")

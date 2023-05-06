import requests
import json
# Replace with your actual API key and email address
api_key = "RPOyuNjXMatzu-L*M1ap"
email = "ye.smertenko@gmail.com"

# Construct the base API URL
base_api_url = f"https://api.acleddata.com/acled/read?key={api_key}&email={email}"

# Define your query filters, limit, and pagination
query_filters = {
    'event_date': '2023-03-01|2023-05-01',
    'event_date_where': 'BETWEEN',
    'country': 'Ukraine',
    'event_type': 'Explosions/Remote violence:OR:event_type=Violence against civilians',
    'fields': 'event_date|event_type|sub_event_type|actor1|actor2|admin1|admin2|admin3|location|latitude|longitude|notes|fatalities|timestamp'
}

# Construct the API URL with query filters
api_url = base_api_url + \
    ''.join([f'&{k}={v}' for k, v in query_filters.items()])

response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Save the data as a JSON file
    with open('acleddata.json', 'w') as json_file:
        json.dump(data, json_file)

    print("Data saved to 'acleddata.json'.")

else:
    print(f"Request failed with status code {response.status_code}.")

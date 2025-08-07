#https://ruthussanketh.medium.com/wikipedia-user-data-extraction-and-preprocessing-45af30642067
#https://www.mediawiki.org/wiki/API:Userinfo
import requests
import json
S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"
u  = "Editor540"
PARAMS = {
        "action": "query",
        "format": "json",
        "list": "users",
        "ususers": u,#"Editor540","Citation bot","24.164.205.69"
        "usprop": "blockinfo|editcount|groups|groupmemberships",
        #"uiprop":"blockinfo|editcount|groupmemberships|registrationdate|groups",
    }

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

print(DATA)
file_path = "output.json"

# Write JSON data to the file
with open(file_path, 'w') as file:
    json.dump(DATA, file)

print(f"JSON data saved to {file_path}")

import requests
import json

BASE_URL = "https://api.coverage.cms.gov"

def get_license_token():
    """Step 1: Get a bearer token from the license agreement endpoint."""
    url = f"{BASE_URL}/v1/metadata/license-agreement"
    resp = requests.get(url)
    print(f"[license-agreement] status: {resp.status_code}")
    body = resp.json()

    print(f"[license-agreement] top-level keys: {list(body.keys())}")
    if "meta" in body:
        print(f"[license-agreement] meta keys: {list(body['meta'].keys())}")
    if "data" in body:
        print(f"[license-agreement] data content:\n{json.dumps(body['data'], indent=2)}")

    resp.raise_for_status()
    return body

def get_lcd(token, lcd_id="L33686"):
    """Step 2: Use the token to fetch one LCD's detail record."""
    url = f"{BASE_URL}/v1/data/lcd/"
    headers = {"Authorization": f"Bearer {token}"}
    # API requires the param named "lcdid" (not "id") and the numeric
    # ID only, without the leading "L" prefix used in the display ID.
    numeric_id = lcd_id.lstrip("Ll")
    params = {"lcdid": numeric_id}
    resp = requests.get(url, headers=headers, params=params)
    print(f"\n[data/lcd] status: {resp.status_code}")
    print(f"[data/lcd] raw body (first 3000 chars):\n{resp.text[:3000]}\n")
    return resp

if __name__ == "__main__":
    license_data = get_license_token()

    data = license_data.get("data", {})
    token = None
    if isinstance(data, list):
        # Confirmed shape: a list of dicts, with a capitalized "Token" key
        for item in data:
            if isinstance(item, dict) and "Token" in item:
                token = item["Token"]
                print(f"\nFound token under data[list item].'Token'")
                break
    elif isinstance(data, dict):
        for key in ("token", "access_token", "bearer_token", "licenseToken", "license_token", "Token"):
            if key in data:
                token = data[key]
                print(f"\nFound token under data.'{key}'")
                break
    elif isinstance(data, str):
        # sometimes APIs just return the token as a raw string in "data"
        token = data
        print(f"\n'data' appears to be the token itself (a string).")

    if token is None:
        print("\nStill couldn't auto-detect the token. The 'data content' printed above")
        print("should show its actual shape — tell me what you see there.")
    else:
        get_lcd(token)
import requests
import json
import os

BASE_URL = "https://api.coverage.cms.gov"
CALIFORNIA_STATE_ID = 6  # "California - Entire State" per /v1/metadata/states/


def get_license_token():
    url = f"{BASE_URL}/v1/metadata/license-agreement"
    resp = requests.get(url)
    body = resp.json()
    resp.raise_for_status()

    data = body.get("data", {})
    token = None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "Token" in item:
                token = item["Token"]
                break
    return token


def get_dme_lcd_list(token):
    """Confirmed real params: state_id (int), status (str: A/R/F).
    Filters to California, Active, and contractor_name_type containing 'DME MAC'."""
    url = f"{BASE_URL}/v1/reports/local-coverage-final-lcds/"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"state_id": CALIFORNIA_STATE_ID, "status": "A"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    all_rows = resp.json().get("data", [])

    dme_rows = [r for r in all_rows if "DME MAC" in r.get("contractor_name_type", "")]
    return dme_rows


if __name__ == "__main__":
    token = get_license_token()
    if token is None:
        print("Could not get token.")
    else:
        dme_rows = get_dme_lcd_list(token)
        print(f"Found {len(dme_rows)} DME LCDs for California (Jurisdiction D).")

        os.makedirs("data/raw", exist_ok=True)
        out_path = "data/raw/lcd_id_list.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dme_rows, f, indent=2)
        print(f"Saved to {out_path}")

        print("\nFirst 5 entries:")
        for row in dme_rows[:5]:
            print(f"  {row['document_display_id']} (id={row['document_id']}, v={row['document_version']}) - {row['title']}")

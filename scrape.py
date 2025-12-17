# Architecture Concept: Scraper for Rekosistem/Mallsampah endpoints
import requests


def get_real_locations(provider="rekosistem"):
    # Target: Hidden API endpoints used by mobile/web maps
    endpoints = {
        "rekosistem": "https://api.rekosistem.com/v1/waste-stations",
        "mallsampah": "https://api.mallsampah.com/v2/drop-points"
    }

    headers = {"User-Agent": "Mozilla/5.0", "Authorization": "Bearer <TOKEN_IF_REQUIRED>"}

    response = requests.get(endpoints[provider], headers=headers)
    if response.status_code == 200:
        return response.json()['data']  # Extract lat, lon, material_type
    return []

# Actionable: Integrate these JSON structures into your SAMPLE_LOCATIONS dictionary.
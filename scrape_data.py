import requests
import json
import concurrent.futures
from tqdm.contrib.concurrent import thread_map
from dotenv import load_dotenv
import pandas as pd
import os
from thefuzz import process, fuzz


load_dotenv()

x_api_key = os.getenv("X_API_KEY")
if x_api_key is None:
    raise ValueError("X_API_KEY environment variable is missing!")

def scrape_transparentia_data(output_filename):
    base_url = "https://transparentia.newtral.es/api/advanced-search"

    headers = {
        'accept': 'application/json, text/plain, */*',
        'referer': 'https://transparentia.newtral.es/busqueda-avanzada',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'x-api-key': x_api_key,
    }

    base_params = {
        'name': '',
        'salaryRange[]': [0, -1],
        'salaryType': 'annualSalary',
        'inactive': 'false'
    }

    all_results = []
    MAX_WORKERS = 10

    def fetch_page(page_number):
        """Fetches a single page of data."""
        params = base_params.copy()
        params['page'] = page_number

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            return data.get("data", {}).get("results", [])
        else:
            print(f"\nWarning: Page {page_number} failed with status {response.status_code}")
            return None

    print("Fetching page 1 to get total page count...")
    page_1_params = base_params.copy()
    page_1_params['page'] = 1

    response = requests.get(base_url, headers=headers, params=page_1_params)

    if response.status_code != 200:
        print(f"Failed to fetch page 1. Status: {response.status_code}")
        return

    data = response.json()
    total_pages = data.get("data", {}).get("pages", 1)
    page_1_results = data.get("data", {}).get("results", [])

    if page_1_results:
        all_results.extend(page_1_results)

    print(f"Discovered {total_pages} total pages. Fetching all pages in parallel...")

    pages_to_fetch = range(1, total_pages + 1)

    # This will return a list of lists (or Nones for failures)
    results_list = thread_map(
        fetch_page,
        pages_to_fetch,
        max_workers=MAX_WORKERS,
        desc="Fetching pages"
    )

    all_results = [item for sublist in results_list if sublist is not None for item in sublist]

    print(f"\nScraping complete.")
    print(f"Total results gathered: {len(all_results)}")
    print(f"Saving all results to {output_filename}...")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"Successfully saved data to temp file {output_filename}.")

def extract_name(cell):
    if isinstance(cell, dict):
        return cell.get('name')
    return None

def extract_slug(cell):
    if isinstance(cell, dict):
        return cell.get('slug')
    return None

def clean_json(file_path, output):
    print(f"Cleaning JSON and saving to {output}")
    df = pd.read_json(file_path)

    df['spatial_name'] = df['spatial'].apply(extract_name)
    df['role_name'] = df['role'].apply(extract_name)
    df['member_of_name'] = df['currentMemberOf'].apply(extract_name)
    df['affiliation_slug'] = df['affiliation'].apply(extract_slug)
    df['affiliation_slug'] = df['affiliation_slug'].replace('pp', 'partido-popular')

    final_columns = [
        "name",
        "currentAnnualSalary",
        "currentMonthlySalary",
        "active",
        "jobTitle",
        "gender",
        "spatial_name",
        "role_name",
        "affiliation_slug",
        "member_of_name"
    ]
    df_clean = df[final_columns]
    df_clean.to_csv(output)
    os.remove(file_path)
    print(f"Successefully processed and saved data to {output}")

def update_csv_with_matches(source_csv, target_csv, output_csv=None, threshold=90):
    print("Using fuzzy matching to find town halls in external dataset")
    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv, sep=";")
    
    
    if 'member_of_name' in source_df.columns:
        source_df['municipio'] = source_df['member_of_name'].str.replace(r'Ayuntamiento De\s*', '', regex=True)
        source_df['municipio'] = source_df['municipio'].apply(lambda x: x.split('/')[0] if '/' in str(x) else x)
    
    source_cities = source_df['municipio'].dropna().unique().tolist()
    target_cities = target_df['Official Name Municipality'].tolist()
    
    city_mapping = {}
    for source_city in source_cities:
        match, score = process.extractOne(source_city, target_cities, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            city_mapping[source_city] = match
        

    updated_rows = 0
    for idx, row in source_df.iterrows():
        if pd.notna(row.get('municipio')) and row['municipio'] in city_mapping:
            matched_name = f"Ayuntamiento De {city_mapping[row['municipio']]}/ES"
            source_df.at[idx, 'member_of_name'] = matched_name
            updated_rows += 1
    
    if output_csv is None:
        output_csv = source_csv
    
    source_df.to_csv(output_csv, index=False)
    
    print(f"Updated {updated_rows} rows with matches above {threshold}% similarity")
    print(f"Updated file saved as: {output_csv}")
    return updated_rows

if __name__ == "__main__":
    scrape_transparentia_data("temp.json")
    clean_json("temp.json", "cleaned.csv")
    rows_updated = update_csv_with_matches("cleaned.csv", "georef-spain-municipio.csv", threshold=90)

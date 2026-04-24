import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from datetime import datetime

class IMDLiveScraper:
    def __init__(self):
        # IMD main warnings portal
        self.url = "https://mausam.imd.gov.in/responsive/all_india_forcast_bulletin.php"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

    def fetch_warnings(self):
        print(f"Fetching live data from IMD: {self.url}...")
        try:
            response = requests.get(self.url, headers=self.headers, verify=False, timeout=10)
            response.raise_for_status()
            return self._parse_html(response.text)
            
        except Exception as e:
            print(f"Failed to fetch IMD data: {e}")
            return []

    def _parse_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        warnings = []
        
        # Look for the warnings section
        warning_keywords = ['heavy', 'rainfall', 'cyclone', 'thunderstorm', 'squall', 'fog', 'heat wave']
        
        for p in soup.find_all(['p', 'li']):
            text = p.get_text(strip=True).lower()
            
            # If the paragraph mentions a warning keyword
            if any(k in text for k in warning_keywords) and ('over' in text or 'at' in text):
                
                # Assign a severity based on keywords
                severity = "Medium"
                risk_score = 0.5
                if 'very heavy' in text or 'extremely heavy' in text or 'cyclone' in text:
                    severity = "High (Red Alert)"
                    risk_score = 0.9
                elif 'heavy' in text or 'severe' in text:
                    severity = "High (Orange Alert)"
                    risk_score = 0.7
                    
                warnings.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Warning_Text": text.capitalize(),
                    "Severity_Label": severity,
                    "Disruption_Risk": risk_score
                })
                
        return warnings

    def save_to_csv(self, warnings, output_path="data/processed/imd_live_alerts.csv"):
        if not warnings:
            print("No warnings found to save.")
            return
            
        df = pd.DataFrame(warnings)
        
        # Append if exists, else write new
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)
            
        print(f"Successfully saved {len(warnings)} live IMD warnings to {output_path}")
        return df


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    scraper = IMDLiveScraper()
    live_warnings = scraper.fetch_warnings()
    
    if live_warnings:
        print("\n--- LIVE IMD WARNINGS DETECTED ---")
        for i, w in enumerate(live_warnings[:5]):  # Show top 5
            print(f"[{w['Severity_Label']}] Risk {w['Disruption_Risk']}: {w['Warning_Text'][:100]}...")
        
        # Save for ML pipeline
        output_file = "../data/processed/imd_live_alerts.csv"
        if not os.path.exists("../data"):
            output_file = "data/processed/imd_live_alerts.csv"
            
        scraper.save_to_csv(live_warnings, output_file)
    else:
        print("No severe warnings detected on the IMD portal right now.")

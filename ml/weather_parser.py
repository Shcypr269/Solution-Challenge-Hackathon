import pdfplumber
import pandas as pd
import re
import os

def parse_imd_rainfall_pdf(pdf_path: str, output_csv: str):

    print(f"Parsing IMD PDF: {pdf_path}")

    records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: continue

            lines = text.split('\n')
            for line in lines:
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue

                name_parts = []
                num_parts = []
                for p in parts[1:]:
                    if p.replace('.', '', 1).isdigit() or p.replace('-', '', 1).replace('%', '', 1).isdigit() or p == '*' or p in ['LE', 'E', 'N', 'D', 'LD', 'NR']:
                        num_parts.append(p)
                    else:
                        name_parts.append(p)

                name = " ".join(name_parts)

                if not name:
                    continue

                daily_cat = "NR"
                period_cat = "NR"

                for i, p in enumerate(num_parts):
                    if p in ['LE', 'E', 'N', 'D', 'LD', 'NR']:
                        if daily_cat == "NR":
                            daily_cat = p
                        else:
                            period_cat = p

                risk_level = "NONE"
                if daily_cat == "LE" or period_cat == "LE":
                    risk_level = "HIGH (Flood Risk)"
                elif daily_cat == "E" or period_cat == "E":
                    risk_level = "MEDIUM (Heavy Rain)"
                elif daily_cat == "LD" or period_cat == "LD":
                    risk_level = "LOW (Drought/Heat)"

                records.append({
                    "Location": name,
                    "Daily_Category": daily_cat,
                    "Period_Category": period_cat,
                    "Weather_Risk": risk_level
                })

    delivery_dataframe = pd.DataFrame(records)
    print(f"Extracted {len(delivery_dataframe)} district records.")

    disruptions = delivery_dataframe[delivery_dataframe['Weather_Risk'].isin(["HIGH (Flood Risk)", "MEDIUM (Heavy Rain)"])]
    print(f"Found {len(disruptions)} districts with active weather risks (Excess/Large Excess Rain).")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    delivery_dataframe.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    return delivery_dataframe

if __name__ == "__main__":
    pdf_path = "../DataSet Download/DISTRICT_RAINFALL_DISTRIBUTION_COUNTRY_INDIA_cd.pdf"
    output_csv = "../data/processed/imd_district_rainfall.csv"

    if not os.path.exists(pdf_path):
        pdf_path = "DataSet Download/DISTRICT_RAINFALL_DISTRIBUTION_COUNTRY_INDIA_cd.pdf"
        output_csv = "data/processed/imd_district_rainfall.csv"

    parse_imd_rainfall_pdf(pdf_path, output_csv)

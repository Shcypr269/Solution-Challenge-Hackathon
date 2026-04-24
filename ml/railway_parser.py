
import pandas as pd
import os

def process_railway_data(loading_csv: str, revenue_csv: str, output_csv: str):
    print("Processing official Indian Railways freight data...")

    df_load = pd.read_csv(loading_csv)
    df_rev = pd.read_csv(revenue_csv)

    df_load.columns = ['Year', 'Total_Loading_MT']
    df_rev.columns = ['Year', 'Total_Loading_MT_2', 'Freight_Revenue_Cr']

    delivery_dataframe = pd.merge(df_load, df_rev[['Year', 'Freight_Revenue_Cr']], on='Year', how='inner')

    delivery_dataframe['Year_Start'] = delivery_dataframe['Year'].str.split('-').str[0].astype(int)
    delivery_dataframe = delivery_dataframe.sort_values('Year_Start').reset_index(drop=True)

    avg_haul_km = 600

    delivery_dataframe['Revenue_Rs'] = delivery_dataframe['Freight_Revenue_Cr'] * 10_000_000
    delivery_dataframe['Loading_Tonnes'] = delivery_dataframe['Total_Loading_MT'] * 1_000_000

    delivery_dataframe['Cost_Per_Tonne_Rs'] = delivery_dataframe['Revenue_Rs'] / delivery_dataframe['Loading_Tonnes']

    delivery_dataframe['Cost_Per_Tonne_Km_Rs'] = delivery_dataframe['Cost_Per_Tonne_Rs'] / avg_haul_km

    print("\n--- Indian Railways Freight Metrics ---")
    for _, row in delivery_dataframe.iterrows():
        print(f"Year {row['Year']}:")
        print(f"  Loading: {row['Total_Loading_MT']} Million Tonnes")
        print(f"  Revenue: {row['Freight_Revenue_Cr']} Crore Rs")
        print(f"  Calculated Cost: Rs {row['Cost_Per_Tonne_Km_Rs']:.2f} per Tonne-Km")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    delivery_dataframe.to_csv(output_csv, index=False)
    print(f"\nSaved calibrated rail metrics to {output_csv}")

    return delivery_dataframe

if __name__ == "__main__":
    loading_file = "../DataSet Download/RS_Session_260_AU_277_B.csv"
    revenue_file = "../DataSet Download/RS_Session_267_AU_3956_A_to_D.csv"
    output_file = "../data/processed/railway_freight_metrics.csv"

    if not os.path.exists(loading_file):
        loading_file = "DataSet Download/RS_Session_260_AU_277_B.csv"
        revenue_file = "DataSet Download/RS_Session_267_AU_3956_A_to_D.csv"
        output_file = "data/processed/railway_freight_metrics.csv"

    process_railway_data(loading_file, revenue_file, output_file)

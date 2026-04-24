import os

def download_lade():
    from huggingface_hub import hf_hub_download
    
    os.makedirs("data/raw/lade", exist_ok=True)
    
    target = "data/raw/lade/delivery_five_cities.csv"
    if os.path.exists(target):
        print(f"Already exists: {target}")
        return target
    
    print("Downloading LaDe-D delivery dataset (~136MB)...")
    print("Source: https://huggingface.co/datasets/Cainiao-AI/LaDe")
    
    path = hf_hub_download(
        repo_id="Cainiao-AI/LaDe",
        filename="delivery_five_cities.csv",
        repo_type="dataset",
        local_dir="data/raw/lade"
    )
    
    print(f"Downloaded to: {path}")
    
    # Quick preview
    import pandas as pd
    delivery_dataframe = pd.read_csv(target, nrows=5)
    print(f"\nColumns ({len(delivery_dataframe.columns)}): {list(delivery_dataframe.columns)}")
    print(delivery_dataframe.head())
    
    # Row count estimate
    import subprocess
    result = subprocess.run(
        ["powershell", "-Command", f"(Get-Content '{target}' | Measure-Object -Line).Lines"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"\nTotal rows: ~{result.stdout.strip()}")
    
    return target

if __name__ == "__main__":
    download_lade()

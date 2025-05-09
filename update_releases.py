from github import Github
from datasets import load_dataset, Dataset
import pandas as pd
from datetime import datetime
from huggingface_hub import login
import os

def get_latest_releases(repo_owner, repo_name, count=10):
    # Create a GitHub instance
    g = Github(os.getenv('GH_TOKEN')) if os.getenv('GH_TOKEN') else Github()  # Use token if available
    
    try:
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        releases = repo.get_releases()
        
        latest_releases = []
        for i, release in enumerate(releases):
            if i >= count:
                break
            latest_releases.append({
                'repo_owner': repo_owner,
                'repo_name': repo_name,
                'tag_name': release.tag_name,
                'name': release.title,
                'published_at': release.created_at.isoformat(),
                'body': release.body,
                'last_updated': datetime.now().isoformat()
            })
        
        return latest_releases
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def update_hf_dataset(repo_owner, repo_name, dataset_name, count=10):
    # Authenticate with Hugging Face
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token=hf_token)

    # Get latest releases
    new_releases = get_latest_releases(repo_owner, repo_name, count)
    if not new_releases:
        return
        
    # Load existing dataset
    try:
        dataset = load_dataset(dataset_name)
        df_existing = dataset['train'].to_pandas()
    except:
        df_existing = pd.DataFrame()

    # Create DataFrame from new releases
    df_new = pd.DataFrame(new_releases)
    
    # Filter out releases that already exist in the dataset
    if not df_existing.empty:
        existing_tags = set(df_existing['tag_name'])
        df_new = df_new[~df_new['tag_name'].isin(existing_tags)]
    
    # Append new releases if any
    if not df_new.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Convert to Hugging Face dataset and push
        new_dataset = Dataset.from_pandas(df_combined)
        new_dataset.push_to_hub(dataset_name)
        print(f"Added {len(df_new)} new releases to the dataset")
    else:
        print("No new releases to add")

if __name__ == "__main__":
    # Configuration
    REPO_OWNER = "huggingface"
    REPO_NAME = "transformers"
    DATASET_NAME = "reach-vb/transformers-releases"  # Replace with your dataset name
    
    update_hf_dataset(REPO_OWNER, REPO_NAME, DATASET_NAME)

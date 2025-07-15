import pandas as pd
import json
import logging
import os

def merge_datasets():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mumu_path = os.path.join(base_dir, 'MuMu', 'mb_albums', 'musicbrainz_album_final_super_genres.csv')
    msdi_path = os.path.join(base_dir, 'MSD-I', 'msdi_album_final_super_genres.csv')
    output_path = os.path.join(base_dir, 'merged_dataset_mumu_msdi_final.csv')
    
    # Read datasets
    logging.info("Reading MuMu dataset...")
    mumu_df = pd.read_csv(mumu_path) 
    logging.info("Reading MSD-I dataset...")
    msdi_df = pd.read_csv(msdi_path)
    
    
    # Merge datasets based on common columns
    logging.info("Merging datasets...")
    merged_df = pd.concat([mumu_df, msdi_df], ignore_index=True)
    print(len(merged_df))

    merged_df.drop_duplicates(subset=['album_group_mbid'], inplace=True)
    print(len(merged_df))
    
    # Save merged dataset
    logging.info("Saving merged dataset...")
    merged_df.to_csv(output_path, index=False)
    logging.info(f"Merged dataset saved to {output_path}")
    
    return merged_df


def add_complexity_score_msdi_mumu():
    old_df = pd.read_csv("./old_merged_dataset_with_complexity.csv")
    new_df = pd.read_csv("./merged_dataset_mumu_msdi_final.csv")
    # Merge the old and new DataFrames on 'album_group_mbid'
    merged_df = new_df.merge(old_df[['album_group_mbid', 'permutation_entropy', 'statistical_complexity']], 
                            on='album_group_mbid', 
                            how='left')

    # Rename the columns
    merged_df.rename(columns={
        'permutation_entropy': 'entropy',
        'statistical_complexity': 'complexity'
    }, inplace=True)

    # Save the updated DataFrame
    merged_df.to_csv("mumu_msdi_final_with_complexity.csv", index=False)


def merge_image_ulrs():
    new_df = pd.read_csv("./merged_dataset_mumu_msdi_final.csv")
    # Merge the image URLs
    new_df['image_url'] = new_df['amazon_image_url'].combine_first(new_df['image_url'])

    # Optionally, drop the original columns if no longer needed
    new_df.drop(columns=['amazon_image_url'], inplace=True)

    # Save the updated DataFrame with merged image URLs
    new_df.to_csv("./merged_dataset_mumu_msdi_final.csv", index=False)


def merge_dataset_complexity_score():
    df = pd.read_csv("./mumu_msdi_final_with_complexity.csv")
    df_billboard = pd.read_csv("./billboard_album_final_with_complexity.csv")
    df_billboard.drop(columns=["image","isNew","lastPos","peakPos","rank","title","weeks","image_url","year","genre"], inplace=True)
    merged_df = pd.concat([df, df_billboard], ignore_index=True)
    merged_df.to_csv("./all_album_final_with_complexity.csv", index=False)


def check_missing_images():
    # Load the dataset
    df = pd.read_csv("./merged_dataset_mumu_msdi_final.csv")
    
    # Get the list of downloaded files
    downloaded_files = {filename.rsplit('.', 1)[0] for filename in os.listdir('./img')}
    
    
    # Check for missing album_group_mbid
    missing_mbids = df[~df['album_group_mbid'].isin(downloaded_files)]['album_group_mbid'].tolist()
    print(len(missing_mbids))
    # Print the results
    print(f"Missing album_group_mbid in dataset/img: {len(missing_mbids)}")
    for mbid in missing_mbids:
        print(mbid)

if __name__ == "__main__":
    # merge_datasets()
    # merge_image_ulrs()
    # add_complexity_score_msdi_mumu()
    # merge_dataset_complexity_score()
    check_missing_images()
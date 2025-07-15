import pandas as pd
import os

def clean_dataset(df):
    # df = df.dropna(subset=['complexity_overall_score'])
    df = df.drop_duplicates(subset=['album_group_mbid'])
    df_billboard = pd.read_csv('./results/billboard_album_with_complexity_cleaned.csv')
    df = df[~df['album_group_mbid'].isin(df_billboard['album_group_mbid'])]
    print(len(df) + len(df_billboard))
    return df

def duplicates_billboard():
    df = pd.read_csv('./results/billboard_album_with_complexity_cleaned_with_gemini_genres.csv')
    print(len(df))
    df = df.drop_duplicates(subset=['album_group_mbid'])
    print (len(df))
    df.to_csv('./results/billboard_album_with_complexity_cleaned_with_gemini_genres.csv', index=False)

def check_no_date():
    # df = pd.read_csv('./data/billboard_album_final_with_labels.csv')
    df = pd.read_csv('./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv')
    # df = pd.read_csv('./results/mumu_msdi_with_complexity.csv')
    # df = pd.read_csv('../dataset/MSD-I/msdi_album_details_with_genres_no_duplicates.csv')
    # df = pd.read_csv("../dataset/MuMu/mb_albums/musicbrainz_album_final_super_genres.csv")
    print(len(df))
    df  = df.drop_duplicates(subset=['album_group_mbid'])
    print(len(df))
    dfno_date = df[df['first_release_date_group'].isna()]
    print(len(dfno_date))
    df = df.dropna(subset=['first_release_date_group'])
    print(len(df))
    # df = df[['album_group_mbid','artist_name','album_group_title']]
    # df = df.dropna(subset=['artist_name'])
    # df = df.dropna(subset=['album_group_title'])
    # df.to_csv('../dataset/MuMu/mb_albums/musicbrainz_album_final_super_genres.csv', index=False)

def check_no_complexity():
    df = pd.read_csv('./results/mumu_msdi_with_complexity.csv')
    print(len(df))
    df = df[df['complexity_overall_score'].isna()]
    print(len(df))

def check_0_complexity():
    df = pd.read_csv('./results/mumu_msdi_with_complexity.csv')
    print(len(df))
    df = df[df['complexity_overall_score'] == 0]
    print(len(df))

def split_dataset(input_file='./data/merged_dataset_mumu_msdi_final_cleaned.csv', chunk_size=10000, output_dir='./data/chunks'):
    """
    Split a large CSV file into smaller chunks of specified size
    
    Args:
        input_file (str): Path to the input CSV file
        chunk_size (int): Number of rows per chunk (default: 10000)
        output_dir (str): Directory to save the chunk files
    """
    print(f"Loading dataset: {input_file}")
    
    # Load the dataset
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    print(f"Total rows: {total_rows:,}")
    print(f"Chunk size: {chunk_size:,}")
    
    # Calculate number of chunks
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
    print(f"Will create {num_chunks} chunks")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Split the dataset into chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_filename = f"mumu_msdi_chunk_{i+1:02d}_rows_{start_idx+1}-{end_idx}.csv"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        chunk_df.to_csv(chunk_path, index=False)
        print(f"  Created {chunk_filename}: {len(chunk_df):,} rows")
    
    print(f"\n✅ Dataset split into {num_chunks} chunks in {output_dir}/")
    return [os.path.join(output_dir, f"mumu_msdi_chunk_{i+1:02d}_rows_{i*chunk_size+1}-{min((i+1)*chunk_size, total_rows)}.csv") 
            for i in range(num_chunks)]


def _get_no_genres_albums(df):
    df = df[df['genres'] == '[]']
    df = df[['album_group_mbid', 'artist', 'album_group_title', 'first_release_date_group']]
    df2 = df[['artist', 'album_group_title', 'first_release_date_group']]
    df2.to_csv('./data/billboard_album_no_genres_only_title.csv', index=False)
    return df


def merge_dataset_mumu_msdi():
    df4 = pd.read_csv('./results/mumu_msdi_chunk_04_rows_30001-35899_with_complexity.csv')
    df3 = pd.read_csv('./results/mumu_msdi_chunk_03_rows_20001-30000_with_complexity.csv')
    df2 = pd.read_csv('./results/mumu_msdi_chunk_02_rows_10001-20000_with_complexity.csv')
    df = pd.read_csv('./results/mumu_msdi_chunk_01_rows_1-10000_with_complexity.csv')
    df_final = pd.concat([df, df2, df3, df4])
    df_final.to_csv('./data/mumu_msdi_with_complexity.csv', index=False)

def add_gemini_genres():
    df = pd.read_csv('./results/billboard_album_with_complexity_no_duplicates.csv')
    df_gemini = pd.read_csv("./data/gemini/genre_prediction/billboard_albums_with_genres_gemini.csv")
    
    df_gemini = df_gemini[['album_group_mbid', 'genres', 'sure']]
    
    # Filter out false values - handle different possible formats
    df_gemini = df_gemini[df_gemini['sure'] != False]
    df_gemini = df_gemini[df_gemini['sure'] != 'False']
    df_gemini = df_gemini[df_gemini['sure'] != 'false']
    df_gemini = df_gemini[df_gemini['sure'] != 0]
    df_gemini = df_gemini[df_gemini['sure'].notna()]
    
    # Rename the gemini genres column to avoid conflicts during merge
    df_gemini = df_gemini.rename(columns={'genres': 'gemini_genres'})
    print("Gemini true predictions")
    print(len(df_gemini))
    df = pd.merge(df, df_gemini[['album_group_mbid', 'gemini_genres']], on='album_group_mbid', how='left')
    
    # Fill missing genres with Gemini predictions and set gemini flag
    df['gemini'] = False
    mask = (df['genres'].isna() | (df['genres'] == '[]') | (df['genres'] == '')) & df['gemini_genres'].notna()
    df.loc[mask, 'genres'] = df.loc[mask, 'gemini_genres']
    df.loc[mask, 'gemini'] = True
    
    # Drop the temporary gemini_genres column
    df = df.drop('gemini_genres', axis=1)
    print(len(df[df['gemini'] == True]))
    # df.to_csv('./results/billboard_album_with_complexity_cleaned_with_gemini_genres.csv', index=False)

def main():
    df = pd.read_csv('./data/merged_dataset_mumu_msdi_final.csv')
    df = clean_dataset(df)
    print(len(df))
    df.to_csv('./data/merged_dataset_mumu_msdi_final_cleaned.csv', index=False)

def remove_image_not_found():
    # Load the cleaned dataset
    df = pd.read_csv("./data/billboard_album_final_super_genres.csv")

    # Get the list of MBIDs from the existing images
    existing_mbids = {filename.split('.')[0] for filename in os.listdir("./data/img_all")}

    # Filter the DataFrame to keep only rows where album_group_mbid is in the existing MBIDs
    df_filtered = df[df['album_group_mbid'].isin(existing_mbids)]

    # Save the filtered dataset
    df_filtered.to_csv("./data/billboard_album_final_super_genres_with_images.csv", index=False)

def check_label_found():
    df = pd.read_csv("./data/merged_dataset_mumu_msdi_final_cleaned.csv")
    print(len(df))
    df = df.dropna(subset=['label_name'])
    print(len(df))
    labels=df['label_name'].unique()
    print(len(labels))
    print(labels)

def add_indipendent_major_labels():
    # Load MUMU-MSDI dataset and labels
    df = pd.read_csv("./data/merged_dataset_mumu_msdi_final_cleaned.csv")
    df_labels = pd.read_csv("./record_label/data/gemini/record_label_prediction/record_label_classifications_mumu_msdi.csv")
    
    # Create a mapping dictionary from label_name to is_independent
    label_to_independent = df_labels.apply(
        lambda x: x['classification'].lower() == 'independent', 
        axis=1
    )
    independent_map = dict(zip(df_labels['label_name'], label_to_independent))
    
    # Map the is_independent values using the dictionary
    df['is_independent'] = df['label_name'].map(independent_map)
    
    # Save the updated dataset
    df.to_csv("./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv", index=False)

    # Do the same for Billboard dataset
    df2 = pd.read_csv("./data/billboard_album_final_super_genres_with_images.csv")
    df_labels_billboard = pd.read_csv("./record_label/data/gemini/record_label_prediction/record_label_classifications.csv")
    
    # Create a mapping dictionary for Billboard labels
    label_to_independent_billboard = df_labels_billboard.apply(
        lambda x: x['classification'].lower() == 'independent', 
        axis=1
    )
    independent_map_billboard = dict(zip(df_labels_billboard['label_name'], label_to_independent_billboard))
    
    # Map the is_independent values using the dictionary
    df2['is_independent'] = df2['label_name'].map(independent_map_billboard)
    
    # Save the updated dataset
    df2.to_csv("./data/billboard_album_final_with_labels.csv", index=False)


def drop_no_images(df):
    """
    Drop albums that don't have corresponding images in the img_all folder.
    
    Args:
        df: DataFrame containing album information with album_group_mbid column
        
    Returns:
        DataFrame with only the albums that have corresponding images
    """
    # Get list of image files from the img_all directory
    image_files = os.listdir("./data/img_all")
    
    # Extract the MBIDs from the filenames (remove .jpg extension)
    available_mbids = {os.path.splitext(filename)[0] for filename in image_files}
    
    # Filter the dataframe to keep only rows where album_group_mbid has a corresponding image
    df_with_images = df[df['album_group_mbid'].isin(available_mbids)]
    
    print(f"Original number of albums: {len(df)}")
    print(f"Number of albums with images: {len(df_with_images)}")
    print(f"Removed {len(df) - len(df_with_images)} albums without images")
    
    return df_with_images

def create_unified_csv():
    df = pd.read_csv("./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv")
    df2 = pd.read_csv("./data/billboard_album_final_with_labels.csv")
    # df = pd.read_csv("../dataset/MuMu/mb_albums/musicbrainz_album_final_super_genres.csv")
    # df2 = pd.read_csv("../dataset/MSD-I/msdi_album_details_with_genres_no_duplicates.csv")
    df = pd.concat([df, df2])
    print(1)
    print(len(df))
    df = df.dropna(subset=['album_group_mbid'])
    print(1.1)
    print(len(df))
    df_duplicates = df[df.duplicated(subset=['album_group_mbid'])]
    print("df_duplicates")
    print(len(df_duplicates))
    df = df.drop_duplicates(subset=['album_group_mbid'])
    print(2)
    print(len(df))
    df = df[["album_group_mbid","album_group_title","artist_name", "release_date","release_title","image_url","msdi_image_url", "genres", "label_name","is_independent",]]
    # df = df[["album_group_mbid","album_group_title","artist_name", "release_date","release_title","image_url","msdi_image_url", "genres", "label_name",]]

    df['image_url'] = df['image_url'].combine_first(df['msdi_image_url'])
    df = df.drop(columns=['msdi_image_url'])
    print(3)
    print(len(df))
    df = df.dropna(subset=['release_date'])
    print(4)
    print(len(df))
    # df = df.dropna(subset=['genres'])
    # print(5)
    # print(len(df))
    df_only_labels = df[df['label_name'].notna()]
    print(len(df_only_labels))
    df = drop_no_images(df)
    print("final")
    print(len(df))

    # df.to_csv("./data/unified_album_dataset.csv", index=False)

def remove_duplicates():
    df = pd.read_csv("./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv")
    print(len(df))
    df = df.drop_duplicates(subset=['album_group_mbid'])
    print(len(df))
    # df.to_csv("./data/unified_album_dataset_no_duplicates.csv", index=False)

def remove_duplicates_comparison():
    df = pd.read_csv("./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv")
    df2 = pd.read_csv("./data/billboard_album_final_with_labels.csv")
    df = pd.concat([df, df2])
    df_duplicates = df[df.duplicated(subset=['album_group_mbid'])]
    print(len(df_duplicates))
    df = df.drop_duplicates(subset=['album_group_mbid'])
    print(len(df))



def create_unified_csv_with_complexity(path_to_df):
    df = pd.read_csv(path_to_df)


    billboard_csv_mdlc = pd.read_csv('./results/billboard_album_with_complexity_cleaned_with_gemini_genres.csv')
    mumu_msdi_csv_mdlc = pd.read_csv('./results/mumu_msdi_with_complexity.csv')
    mdlc_df = pd.concat([billboard_csv_mdlc, mumu_msdi_csv_mdlc])
    mdlc_df = mdlc_df[["album_group_mbid","complexity_overall_score"]]
    mdlc_df = mdlc_df.rename(columns={"complexity_overall_score": "mdlc_complexity_score"})
    print("before")
    print(len(mdlc_df))
    mdlc_df = mdlc_df.drop_duplicates(subset=['album_group_mbid'])
    print("after")
    print(len(mdlc_df))
    
    print("\nBefore merge:")
    print(f"Main df rows: {len(df)}")
    print(f"MDLC df rows: {len(mdlc_df)}")
    print(f"Unique album_group_mbid in main df: {df['album_group_mbid'].nunique()}")
    print(f"Unique album_group_mbid in MDLC df: {mdlc_df['album_group_mbid'].nunique()}")
    
    result_df_mdlc = df.merge(mdlc_df, on="album_group_mbid", how="left")
    result_df_mdlc = result_df_mdlc.dropna(subset=['mdlc_complexity_score'])
    
    print("\nAfter merge:")
    print(f"Result df rows: {len(result_df_mdlc)}")
    print(f"Unique album_group_mbid in result: {result_df_mdlc['album_group_mbid'].nunique()}")
    print(f"Rows with MDLC score: {len(result_df_mdlc[result_df_mdlc['mdlc_complexity_score'].notna()])}")
    
    # print(6)
    # print(len(result_df))
    # result_df = result_df.dropna(subset=['mdlc_complexity_score'])
    # print(7)
    # print(len(result_df))

    billboard_df_zip = pd.read_csv("./results/zip/billboard_album_compression.csv")
    mumu_masdi_df_zip = pd.read_csv("./results/zip/mumu_msdi_album_compression.csv")
    zip_df = pd.concat([billboard_df_zip, mumu_masdi_df_zip])
    zip_df = zip_df[["album_group_mbid","compression_ratio"]]
    zip_df = zip_df.drop_duplicates(subset=['album_group_mbid'])
    print("\nBefore merge:")
    print(f"Main df rows: {len(df)}")
    print(f"Zip df rows: {len(zip_df)}")
    print(f"Unique album_group_mbid in main df: {df['album_group_mbid'].nunique()}")
    print(f"Unique album_group_mbid in Zip df: {zip_df['album_group_mbid'].nunique()}")
    
    result_df_zip = df.merge(zip_df, on="album_group_mbid", how="left")
    result_df_zip = result_df_zip.dropna(subset=['compression_ratio'])
    
    print("\nAfter merge:")
    print(f"Result df rows: {len(result_df_zip)}")
    print(f"Unique album_group_mbid in result: {result_df_zip['album_group_mbid'].nunique()}")
    print(f"Rows with compression ratio: {len(result_df_zip[result_df_zip['compression_ratio'].notna()])}")
    # print(6)
    # print(len(result_df))
    # result_df = result_df.dropna(subset=['compression_ratio'])
    # print(7)
    # print(len(result_df))


    billboard_path = pd.read_csv("./results/plane/billboard_album_entr_compl.csv")
    mumu_path = pd.read_csv("./results/plane/mumu_msdi_album_entr_compl.csv")
    entr_compl_df = pd.concat([billboard_path, mumu_path])
    entr_compl_df = entr_compl_df[["album_group_mbid","permutation_entropy","statistical_complexity"]]
    entr_compl_df = entr_compl_df.drop_duplicates(subset=['album_group_mbid'])
    print("\nBefore merge:")
    print(f"Main df rows: {len(df)}")
    print(f"Entr compl df rows: {len(entr_compl_df)}")
    print(f"Unique album_group_mbid in main df: {df['album_group_mbid'].nunique()}")
    print(f"Unique album_group_mbid in Entr compl df: {entr_compl_df['album_group_mbid'].nunique()}")

    result_df_entr_compl = df.merge(entr_compl_df, on="album_group_mbid", how="left")
    result_df_entr_compl = result_df_entr_compl.dropna(subset=['permutation_entropy','statistical_complexity'])
    
    print("\nAfter merge:")
    print(f"Result df rows: {len(result_df_entr_compl)}")
    print(f"Unique album_group_mbid in result: {result_df_entr_compl['album_group_mbid'].nunique()}")
    print(f"Rows with permutation entropy: {len(result_df_entr_compl[result_df_entr_compl['permutation_entropy'].notna()])}")
    print(f"Rows with statistical complexity: {len(result_df_entr_compl[result_df_entr_compl['statistical_complexity'].notna()])}")
    # print(6)
    # print(len(result_df))
    # result_df = result_df.dropna(subset=['permutation_entropy','statistical_complexity'])
    # print(7)
    # print(len(result_df))

    result_df = result_df_mdlc.merge(result_df_zip[['album_group_mbid','compression_ratio']], on="album_group_mbid", how="left")
    print(len(result_df))
    result_df = result_df.merge(result_df_entr_compl[['album_group_mbid','permutation_entropy','statistical_complexity']], on="album_group_mbid", how="left")
    print(len(result_df))
    # result_df.to_csv("./results/unified_album_dataset_with_complexity.csv", index=False)

def checkGeminiGenres():
    df = pd.read_csv("./results/billboard_album_with_complexity_cleaned_with_gemini_genres.csv")
    print(len(df))
    df = df[df['gemini'] == True]
    print(len(df))

def check_date_with_og():
    df = pd.read_csv("./data/unified_album_dataset.csv")

    df_compl = pd.read_csv("./results/unified_album_dataset_with_complexity.csv")
    # df_mumu = pd.read_csv("../dataset/MuMu/mb_albums/musicbrainz_album_final_super_genres.csv")
    # df_msdi = pd.read_csv("../dataset/MSD-I/msdi_album_details_with_genres_no_duplicates.csv")
    # df_billboard = pd.read_csv("./data/billboard_album_final_with_labels.csv")
    # df_merged_mumu_msdi = pd.read_csv("./data/merged_dataset_mumu_msdi_final_cleaned_with_labels.csv")
    # df_mumu = df_mumu[['album_group_mbid', 'first_release_date_group']]
    # df_mumu = df_mumu.dropna(subset=['first_release_date_group'])
    # print(len(df_mumu))
    # df_msdi = df_msdi[['album_group_mbid', 'first_release_date_group']]
    # df_msdi = df_msdi.dropna(subset=['first_release_date_group'])

    # df_mumu = drop_no_images(df_mumu)
    # df_msdi = drop_no_images(df_msdi)

    # df_concat = pd.concat([df_mumu, df_msdi])
    # print(len(df_merged_mumu_msdi))
    # df_no_in_merged = df_merged_mumu_msdi[~df_merged_mumu_msdi['album_group_mbid'].isin(df_concat['album_group_mbid'])]
    # print(len(df_no_in_merged))
    # df_merged_mumu_msdi = df_merged_mumu_msdi.dropna(subset=['first_release_date_group'])
    # print(len(df_merged_mumu_msdi))
    # df_merged_mumu_msdi = df_merged_mumu_msdi.drop_duplicates(subset=['album_group_mbid'])
    # print(len(df_merged_mumu_msdi))
    # df_no_in_merged = df_merged_mumu_msdi[~df_merged_mumu_msdi['album_group_mbid'].isin(df_concat['album_group_mbid'])]
    # print(len(df_no_in_merged))
    # dupl = len(df_concat)-len(df_merged_mumu_msdi)
    # print(dupl)
    # print(len(df_merged_mumu_msdi) + len(df_billboard))
    # print(len(df_concat)+len(df_billboard)-dupl)
    # print(len(df_billboard)-dupl)
    # print(len(df_billboard))
    # df_billboard = df_billboard.drop_duplicates(subset=['album_group_mbid'])
    # print(len(df_billboard))
    # df_uni_new = pd.concat([df_merged_mumu_msdi, df_billboard])
    # print(len(df_uni_new))
    # df_merged_mumu_msdi['year'] = pd.to_datetime(df_merged_mumu_msdi['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    # df_billboard['year'] = pd.to_datetime(df_billboard['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    # df_mumu['year'] = pd.to_datetime(df_mumu['first_release_date_group'], errors='raise', format='mixed', yearfirst=True).dt.year
    # df_msdi['year'] = pd.to_datetime(df_msdi['first_release_date_group'], errors='raise', format='mixed', yearfirst=True).dt.year
    # # Filter out rows with invalid years
    # df_merged_mumu_msdi = df_merged_mumu_msdi.dropna(subset=['year'])
    # df_merged_mumu_msdi['year'] = df_merged_mumu_msdi['year'].astype(int)
    
    # # Filter reasonable year range (e.g., 1950-2025)
    # df_merged_mumu_msdi = df_merged_mumu_msdi[(df_merged_mumu_msdi['year'] >= 1950) & (df_merged_mumu_msdi['year'] <= 2025)]  
    # print(len(df_merged_mumu_msdi))
    # df_billboard['year'] = df_billboard['year'].astype(int)
    # print(len(df_billboard))
    # df_billboard = df_billboard[(df_billboard['year'] >= 1950) & (df_billboard['year'] <= 2025)]  
    # print(len(df_billboard))
    # df_mumu['year'] = df_mumu['year'].astype(int)
    # print(len(df_mumu))
    # df_mumu = df_mumu[(df_mumu['year'] >= 1950) & (df_mumu['year'] <= 2025)]  
    # print(len(df_mumu))


    # df_msdi['year'] = df_msdi['year'].astype(int)
    # print(len(df_msdi))
    # df_msdi = df_msdi[(df_msdi['year'] >= 1950) & (df_msdi['year'] <= 2025)]  
    # print(len(df_msdi))
    # print(len(df_mumu)+len(df_msdi)+len(df_billboard))
    

    # df['year'] = pd.to_datetime(df['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    # df['year'] = df['year'].astype(int)
    # df = df[(df['year'] >= 1950) & (df['year'] <= 2025)]  
    # print(len(df))

    # df_compl = pd.read_csv("./results/unified_album_dataset_with_complexity.csv")
    # print(len(df_compl))
    df_compl['year'] = pd.to_datetime(df_compl['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    df_compl['year'] = df_compl['year'].astype(int)
    df_compl_year = df_compl[(df_compl['year'] >= 1950) & (df_compl['year'] <= 2025)]  
    print(len(df_compl_year))
    print(len(df_compl)-len(df_compl_year))

    df_compl = df_compl[df_compl['album_group_mbid'].isin(df_compl_year['album_group_mbid'])]
    print(len(df_compl))
    df_compl = df_compl.drop(columns=['year'])
    df_compl.to_csv("./results/unified_album_dataset_with_complexity.csv", index=False)


    df['year'] = pd.to_datetime(df['release_date'], errors='raise', format='mixed', yearfirst=True).dt.year
    df['year'] = df['year'].astype(int)
    df_year = df[(df['year'] >= 1950) & (df['year'] <= 2025)]  
    print(len(df_year))
    print(len(df)-len(df_year))

    df = df[df['album_group_mbid'].isin(df_year['album_group_mbid'])]
    print(len(df))
    df = df.drop(columns=['year'])
    df.to_csv("./data/unified_album_dataset.csv", index=False)

    # df_not_in_og = df_concat[~df_concat['album_group_mbid'].isin(df['album_group_mbid'])]
    # print(len(df_billboard),"-", len(df_mumu),"-", len(df_msdi))
    # df_billboard = df_billboard.dropna(subset=['first_release_date_group'])
    # df_billboard = df_billboard[~df_billboard['album_group_mbid'].isin(df_concat['album_group_mbid'])]
    # print(len(df_billboard))
    # print(len(df_concat))
    # print(len(df_not_in_og))
    # df_concat = pd.concat([df_concat, df_billboard])
    # df_concat = df_concat.dropna(subset=['first_release_date_group'])
    # print(len(df_concat))
    # df_concat = df_concat.drop_duplicates(subset=['album_group_mbid'])
    # df_no_in_unified = df[~df['album_group_mbid'].isin(df_concat['album_group_mbid'])]
    # print(len(df_no_in_unified))
    
    # df_not_in_og.to_csv("./data/not_in_og.csv", index=False)

    
    # df_no_date = df[df['first_release_date_group'].isna()]
    # df_no_date = df_no_date[['album_group_mbid', 'first_release_date_group']]
    # print(len(df_no_date))
    # df_no_date = df_no_date.merge(df_mumu, on="album_group_mbid", how="left")
    # print(len(df_no_date))
    # df_no_date = df_no_date.merge(df_msdi, on="album_group_mbid", how="left")
    # print(len(df_no_date))
    # df_no_date = df_no_date.dropna(subset=['first_release_date_group'])
    # print(len(df_no_date))
    
    
if __name__ == '__main__':
    # # Uncomment the line below if you want to run the cleaning first
    # main()
    
    # # Split the cleaned dataset into 10k row chunks
    # print("Splitting cleaned dataset into 10k row chunks...")
    # chunk_files = split_dataset()
    # print(f"Created {len(chunk_files)} chunk files")

    # df = pd.read_csv('./data/billboard_album_final_super_genres.csv')
    # df = _get_no_genres_albums(df)
    # print(len(df))
    # df.to_csv('./data/billboard_album__no_genres.csv', index=False)
    # merge_dataset_mumu_msdi()
    # add_gemini_genres()
    # remove_image_not_found()
    # remove_duplicates()
    # remove_duplicates_comparison()
    # duplicates_billboard()
    # check_no_date()
    # check_no_complexity()
    # check_0_complexity()
    # check_label_found()
    # add_indipendent_major_labels()
    # create_unified_csv()
    create_unified_csv_with_complexity("./data/unified_album_dataset.csv")
    # checkGeminiGenres()
    check_date_with_og()


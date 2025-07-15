import pandas as pd
import os

# Read the CSV file
input_file = './msdi_album_details_with_genres_no_duplicates.csv'
mumu_file = '../MuMu/mb_albums/musicbrainz_album_final_super_genres.csv'
output_file = './msdi_album_details_with_genres_no_duplicates.csv'

# Read the CSV file
print(f"Reading file: {input_file}")
df = pd.read_csv(input_file)

# Print initial shape
print(f"Initial number of rows: {len(df)}")

# Remove duplicates based on album_group_mbid
# Keep the first occurrence of each album_group_mbid
df_no_duplicates = df.drop_duplicates(subset=['album_group_mbid'], keep='first')
# Read the MuMu dataset
print(f"Reading file: {mumu_file}")
df_mumu = pd.read_csv(mumu_file)

# Remove albums from df_no_duplicates that are already in the MuMu dataset
df_no_duplicates = df_no_duplicates[~df_no_duplicates['album_group_mbid'].isin(df_mumu['album_group_mbid'])]


# Print final shape
print(f"Number of rows after removing duplicates: {len(df_no_duplicates)}")
print(f"Number of duplicates removed: {len(df) - len(df_no_duplicates)}")


# Save the result to a new CSV file
df_no_duplicates.to_csv(output_file, index=False)
print(f"Saved deduplicated data to: {output_file}")

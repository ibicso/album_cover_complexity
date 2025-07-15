import pandas as pd
import os

# Read the CSV file
# input_file = './musicbrainz_album_details_with_images.csv'
input_file= './musicbrainz_album_final_super_genres.csv'
output_file = './musicbrainz_album_final_super_genres.csv'

# Read the CSV file
print(f"Reading file: {input_file}")
df = pd.read_csv(input_file)

# Print initial shape
print(f"Initial number of rows: {len(df)}")

# Remove duplicates based on album_group_mbid
# Keep the first occurrence of each album_group_mbid
df_no_duplicates = df.drop_duplicates(subset=['album_group_mbid'], keep='first')

# Print final shape
print(f"Number of rows after removing duplicates: {len(df_no_duplicates)}")
print(f"Number of duplicates removed: {len(df) - len(df_no_duplicates)}")

df_no_duplicates = df_no_duplicates.dropna(subset=['release_date'])
print(len(df_no_duplicates))

# Save the result to a new CSV file
df_no_duplicates.to_csv(output_file, index=False)
# print(f"Saved deduplicated data to: {output_file}")

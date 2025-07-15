import pandas as pd

mumu_single_file = "mb_albums/musicbrainz_album_details_with_genres.csv"
merged_dataset = "../merged_dataset_renamed.csv"
msdi_dataset = "../../dataset/MSD-I/msdi_album_details_with_genres.csv"

# df = pd.read_csv(mumu_single_file)
# df = pd.read_csv(merged_dataset)
df = pd.read_csv(msdi_dataset)

# Split the genres string and explode to get one row per genre
genre_counts = df['genre'].str.split(',').explode().value_counts()

print("\nNumber of albums per genre:")
print(genre_counts)
with open("genre_counts_msdi.txt", "w") as f:
    for genre, count in genre_counts.items():
        f.write(f"{genre}: {count}\n")
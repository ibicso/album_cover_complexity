import pandas as pd
import os
import logging
import re
from typing import List
import matplotlib.pyplot as plt
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_string(s):
    if pd.isna(s):
        return ""
    # Convert to lowercase, remove special characters and extra spaces
    s = str(s).lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def remove_duplicates():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    billboard_path = os.path.join(base_dir, 'billboard', 'filtered_unique_albums.csv')
    merged_path = os.path.join(base_dir, 'merged_dataset_mumu_msdi_final.csv')
    output_path = os.path.join(base_dir, 'billboard', 'billboard_no_duplicates.csv')
    duplicate_info_path = os.path.join(base_dir, 'billboard', 'removed_duplicates_info.csv')
    
    # Read datasets
    logging.info("Reading datasets...")
    billboard_df = pd.read_csv(billboard_path)
    merged_df = pd.read_csv(merged_path, low_memory=False)
    
    initial_count = len(billboard_df)
    logging.info(f"Initial Billboard dataset size: {initial_count} albums")
    
    # Normalize column names
    logging.info("Normalizing strings...")
    billboard_df['artist_name_norm'] = billboard_df['artist'].apply(normalize_string)
    billboard_df['album_title_norm'] = billboard_df['title'].apply(normalize_string)
    merged_df['artist_name_norm'] = merged_df['artist_name'].apply(normalize_string)
    merged_df['album_title_norm'] = merged_df['album_group_title'].apply(normalize_string)
    
    # Find duplicates
    logging.info("Finding duplicates...")
    duplicates = []
    non_duplicate_indices = []
    
    # Create progress counter
    total = len(billboard_df)
    progress_step = max(1, total // 100)  # Show progress every 1%
    
    for idx, billboard_row in billboard_df.iterrows():
        if idx % progress_step == 0:
            logging.info(f"Progress: {idx}/{total} ({(idx/total)*100:.1f}%)")
        
        # Check for exact matches
        exact_matches = merged_df[
            (merged_df['artist_name_norm'] == billboard_row['artist_name_norm']) &
            (merged_df['album_title_norm'] == billboard_row['album_title_norm'])
        ]
        
        if exact_matches.empty:
            non_duplicate_indices.append(idx)
        else:
            for _, merged_row in exact_matches.iterrows():
                duplicates.append({
                    'billboard_artist': billboard_row['artist'],
                    'billboard_album': billboard_row['title'],
                    'billboard_year': billboard_row['year'],
                    'billboard_genre': billboard_row['genre'],
                    'merged_artist': merged_row['artist_name'],
                    'merged_album': merged_row['album_group_title'],
                    'merged_year': merged_row['release_date'],
                    'merged_genres': merged_row['genres']
                })
    
    # Create filtered dataset without duplicates
    filtered_df = billboard_df.iloc[non_duplicate_indices].copy()
    filtered_df = filtered_df.drop(['artist_name_norm', 'album_title_norm'], axis=1)
    
    # Create DataFrame with duplicate information
    duplicates_df = pd.DataFrame(duplicates)
    
    # Save results
    logging.info("Saving results...")
    filtered_df.to_csv(output_path, index=False)
    duplicates_df.to_csv(duplicate_info_path, index=False)
    
    # Print summary
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    
    logging.info("\nDuplicate Removal Summary:")
    logging.info(f"Initial Billboard albums: {initial_count}")
    logging.info(f"Duplicates found and removed: {removed_count}")
    logging.info(f"Final Billboard albums: {final_count}")
    logging.info(f"Percentage of duplicates: {(removed_count/initial_count)*100:.2f}%")
    
    # Print sample of removed duplicates
    if not duplicates_df.empty:
        logging.info("\nSample of removed duplicates (first 5):")
        for _, row in duplicates_df.head().iterrows():
            logging.info(f"\nBillboard: {row['billboard_artist']} - {row['billboard_album']} ({row['billboard_year']})")
            logging.info(f"Merged: {row['merged_artist']} - {row['merged_album']} ({row['merged_year']})")
            logging.info(f"Billboard genre: {row['billboard_genre']}")
            logging.info(f"Merged genres: {row['merged_genres']}")

def create_dataset(progress_folder: str) -> pd.DataFrame:
    """
    Merges all CSV files in the progress folder into a single DataFrame.
    
    Args:
        progress_folder (str): Path to the folder containing the CSV files
        
    Returns:
        pd.DataFrame: Merged DataFrame containing all data from CSV files
    """
    # List all CSV files in the progress folder
    csv_files = [f for f in os.listdir(progress_folder) if f.endswith('.csv')]
    
    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(progress_folder, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates if any
    merged_df = merged_df.drop_duplicates()
    
    return merged_df

def create_not_found_dataset(merged_df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Creates a new DataFrame with only the rows where found_in_musicbrainz is False
    and saves it to a CSV file.
    
    Args:
        merged_df (pd.DataFrame): The merged DataFrame from create_dataset
        output_file (str): Path where to save the new CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing only rows where found_in_musicbrainz is False
    """
    # Filter rows where found_in_musicbrainz is False
    not_found_df = merged_df[merged_df['found_in_musicbrainz'] == False]
    
    # Save to CSV
    not_found_df.to_csv(output_file, index=False)
    
    return not_found_df

def get_unique_genres(merged_df: pd.DataFrame) -> List[str]:
    """
    Returns a list of all unique genres from the dataset.
    
    Args:
        merged_df (pd.DataFrame): The merged DataFrame from create_dataset
        
    Returns:
        List[str]: List of unique genres
    """
    # Get all genres
    all_genres = []
    for genres_str in merged_df['genres'].dropna():
        try:
            # Convert string representation of list to actual list
            genres = eval(genres_str)
            all_genres.extend(genres)
        except:
            continue
    
    # Get unique genres
    unique_genres = sorted(list(set(all_genres)))
    
    return unique_genres

def remove_non_genre_tags(path: str):
    df = pd.read_csv(path)

    music_genres = [
  "abstract hip hop", "acid breaks", "acid house", "acid jazz", "acid rock", 
  "acid techno", "acoustic", "acoustic blues", "acoustic rock", 
  "adult alternative pop rock", "adult contemporary", "african", "african blues", 
  "afro house", "afro-cuban jazz", "afro-jazz", "afrobeat", "afrobeats", 
  "alternative", "alternative country", "alternative dance", "alternative folk", 
  "alternative hip hop", "alternative metal", "alternative pop", "alternative punk", 
  "alternative r&b", "alternative rock", "alté", "ambient", "ambient pop", 
  "ambient techno", "americana", "anarcho-punk", "anatolian rock", 
  "andalusian classical", "anti-folk", "aor", "arabic jazz", "arena rock", 
  "art pop", "art punk", "art rock", "atmospheric black metal", 
  "atmospheric sludge metal", "avant-folk", "avant-garde", "avant-garde jazz", 
  "avant-garde metal", "bachata", "bakersfield sound", "ballet", 
  "ballroom house", "baltimore club", "banda sinaloense", "baroque", "baroque pop", 
  "bassline", "beat music", "beatdown hardcore", "bebop", "bedroom pop", 
  "berlin school", "bhangra", "big band", "big beat", "bitpop", "black metal", 
  "blackened death metal", "blackgaze", "blue-eyed soul", "bluegrass", 
  "bluegrass gospel", "blues", "blues rock", "bolero", "boogie", "boogie rock", 
  "boogie-woogie", "boom bap", "bossa nova", "bounce", "brass band", "breakbeat", 
  "breaks", "british blues", "british folk rock", "british rhythm & blues", 
  "britpop", "bro-country", "broken beat", "brostep", "brutal death metal", 
  "bubblegum bass", "bubblegum pop", "cabaret", "cajun", "calypso", 
  "canzone napoletana", "ccm", "celtic", "celtic new age", "celtic punk", 
  "celtic rock", "central asian throat singing", "chacarera", "chamber folk", 
  "chamber pop", "chanson française", "chicago blues", "chicago drill", 
  "chicano rap", "children's music", "chillout", "chillwave", "chipmunk soul", 
  "chiptune", "choral", "christian", "christian - worship", "christian hip hop", 
  "christian metal", "christian pop", "christian r n b", "christian rap", 
  "christian rock", "christmas", "christmas music", "city pop", 
  "classic pop vocals", "classic rock", "classical", "classical crossover", 
  "close harmony", "cloud rap", "club", "cocktail nation", "coldwave", "comedy", 
  "comedy hip hop", "comedy rock", "complextro", "concerto", "conscious hip hop", 
  "contemporary christian", "contemporary classical", "contemporary country", 
  "contemporary folk", "contemporary gospel", "contemporary jazz", 
  "contemporary r&b", "cool jazz", "corrido", "country", "country and western", 
  "country blues", "country folk", "country gospel", "country pop", "country rap", 
  "country rock", "country soul", "crossover jazz", "crunk", "crunkcore", 
  "crust punk", "cumbia", "dance", "dance-pop", "dance-punk", "dance-rock", 
  "dancehall", "dark ambient", "dark cabaret", "dark electro", "dark folk", 
  "dark jazz", "dark wave", "death 'n' roll", "death industrial", "death metal", 
  "deathcore", "deathgrind", "deathrock", "deconstructed club", "deep house", 
  "deep soul", "delta blues", "descarga", "desert blues", "desert rock", 
  "detroit trap", "digital fusion", "dirty south", "disco", "diva house", 
  "dixieland", "djent", "doo-wop", "doom metal", "downtempo", "dream pop", 
  "drill", "drone", "drum and bass", "drumless hip hop", "dub", "dub poetry", 
  "dubstep", "dungeon synth", "duranguense", "dutch house", "east coast hip hop", 
  "easy listening", "easycore", "ebm", "edm", "electric blues", 
  "electric texas blues", "electro", "electro house", "electro-disco", 
  "electro-industrial", "electroacoustic", "electroclash", "electronic", 
  "electronic rock", "electronica", "electronica dance", "electronicore", 
  "electropop", "electropunk", "electrotango", "emo", "emo pop", "emo rap", 
  "emocore", "ethereal wave", "euro house", "euro-trance", "eurodance", 
  "european folk/pop", "europop", "experimental", "experimental electronic", 
  "experimental hip hop", "experimental rock", "fado", "festival progressive house", 
  "field recording", "filk", "film score", "filmi", "flamenco", "flamenco jazz", 
  "flamenco pop", "folk", "folk metal", "folk pop", "folk punk", "folk rock", 
  "folk-rock", "folk/country", "folk/rock", "folktronica", "freak folk", 
  "free folk", "free improvisation", "free jazz", "freestyle", "french electro", 
  "french house", "funk", "funk / soul", "funk metal", "funk rock", "funktronica", 
  "funky house", "fusion", "future bass", "future garage", "future rave", 
  "futurepop", "g-funk", "gagaku", "gamelan", "gangsta rap", "garage house", 
  "garage punk", "garage rock", "garage rock revival", "geek rock", "ghetto house", 
  "glam", "glam metal", "glam rock", "glitch", "glitch hop", "glitch pop", 
  "gospel", "gospel and religious", "gospel/country", "gothic", "gothic country", 
  "gothic metal", "gothic rock", "gregorian chant", "grime", "grindcore", "griot", 
  "groove metal", "grunge", "grupera", "guaguancó", "guajira", "gulf", 
  "gulf and western", "gypsy jazz", "gypsy punk", "hands up", "hard bop", 
  "hard house", "hard rock", "hard trance", "hardcore hip hop", "hardcore punk", 
  "hawaiian", "heartland rock", "heavy metal", "heavy psych", "highlife", 
  "hill country blues", "hindustani classical", "hip hop", "hip hop rap", 
  "hip hop soul", "hip house", "hip-hop", "hip-hop/rap", "hiphop/rap/r&b", 
  "hiplife", "honky tonk", "honky tonk and outlaw", "hopepunk", "horror punk", 
  "horror synth", "horrorcore", "house", "hyperpop", "hyphy", "hypnagogic pop", 
  "idm", "impressionism", "indeterminacy", "indian classical", "indian pop", 
  "indie", "indie folk", "indie pop", "indie rock", "indie surf", "indietronica", 
  "industrial", "industrial metal", "industrial rock", "instrumental", 
  "instrumental hip hop", "instrumental jazz", "instrumental rock", "irish folk", 
  "isicathamiya", "j-pop", "j-rock", "jam band", "jangle pop", "jazz", 
  "jazz and blues", "jazz blues", "jazz fusion", "jazz pop", "jazz rap", 
  "jazz rock", "jazz vocals", "jazz-funk", "jerk rap", "jersey drill", 
  "jump blues", "jungle", "k-pop", "kawaii metal", "kayōkyoku", "klezmer", 
  "kompa", "korean ballad", "krautrock", "kwaito", "latin", "latin ballad", 
  "latin dance", "latin jazz", "latin pop", "latin rock", "leftfield", 
  "levenslied", "lo-fi", "louisiana blues", "lounge", "lovers rock", 
  "mainstream rock", "mambo", "mandopop", "mariachi", "math rock", "mathcore", 
  "medieval", "medieval rock", "melodic black metal", "melodic death metal", 
  "melodic dubstep", "melodic hardcore", "melodic house", "melodic metalcore", 
  "memphis rap", "merengue", "merenhouse", "merseybeat", "metal", "metalcore", 
  "miami bass", "microtonal classical", "minimalism", "minneapolis sound", 
  "mod", "modal jazz", "modern", "modern blues", "modern classical", 
  "modern creative", "moombahcore", "moombahton", "morna", "motown", "mpb", 
  "music hall", "musical", "musique concrète instrumentale", 
  "musique régionale mexicaine", "neo soul", "neo-progressive rock", 
  "neo-psychedelia", "neo-traditional country", "neoclassical dark wave", 
  "neoclassical metal", "neoclassical new age", "neoclassicism", "neofolk", 
  "neoperreo", "nerdcore", "neue deutsche härte", "new age", "new jack swing", 
  "new orleans blues", "new orleans r&b", "new rave", "new romantic", "new wave", 
  "no wave", "noise", "noise pop", "noise rock", "norteño", 
  "norteño-banda", "nu disco", "nu jazz", "nu metal", "nuevo flamenco", 
  "observational comedy", "oi", "old school hip hop", "old-time", "opera", 
  "operatic pop", "orchestral", "orchestral jazz", "orchestral song", 
  "outlaw country", "pagan folk", "piano blues", "piano rock", "piedmont blues", 
  "plena", "plunderphonics", "poetry", "political hip hop", "pop", 
  "pop and chart", "pop metal", "pop punk", "pop rap", "pop rock", "pop soul", 
  "pop/rock", "pop/rock/indie/electronic", "post-bop", "post-britpop", 
  "post-grunge", "post-hardcore", "post-industrial", "post-metal", "post-punk", 
  "post-punk revival", "post-rock", "power metal", "power pop", "praise & worship", 
  "production music", "progressive", "progressive bluegrass", "progressive country", 
  "progressive electronic", "progressive folk", "progressive house", 
  "progressive metal", "progressive pop", "progressive rock", "progressive trance", 
  "proto-punk", "psychedelic", "psychedelic folk", "psychedelic pop", 
  "psychedelic rock", "psychedelic soul", "psychobilly", "pub rock", "punk", 
  "punk blues", "punk rock", "qawwali", "quiet storm", "r b", "r b soul", "r&b", 
  "r&b/soul", "rage", "ragga", "ragga hip-hop", "ragtime", "ranchera", "rap", 
  "rap and hip hop", "rap and hip-hop", "rap hip-hop r b", "rap metal", 
  "rap rock", "rap/hip hop", "rap/hip-hop", "red dirt", "reggae", "reggae rock", 
  "reggae-pop", "reggaeton", "regional mexicano", "regueton", "religious", 
  "remix", "renaissance", "rhumba", "rhythm & blues", "ritual ambient", 
  "rnb/swing", "rock", "rock and indie", "rock and roll", "rock en espanol", 
  "rock musical", "rock opera", "rock pop", "rock/soul", "rockabilly", 
  "rocksteady", "romantic classical", "romanticas", "roots reggae", "roots rock", 
  "roots/world/jazz/soul", "rumba", "rumba catalana", "sacred", "salsa", "samba", 
  "sample drill", "score", "screamo", "shoegaze", "sierreño", "singer-songwriter", 
  "ska", "ska punk", "skate punk", "sketch comedy", "slack-key guitar", 
  "slowcore", "sludge metal", "smooth jazz", "smooth soul", "snap", "soca", 
  "soft rock", "son cubano", "sophisti-pop", "soukous", "soul", "soul and reggae", 
  "soul blues", "soul jazz", "sound collage", "soundtrack", "southern gospel", 
  "southern hip hop", "southern rock", "southern soul", "space ambient", 
  "space rock", "speech", "speed metal", "spiritual jazz", "spoken word", 
  "standup comedy", "stomp and holler", "stoner metal", "stoner rock", 
  "street punk", "stride", "surf", "surf rock", "swing", "swing revival", 
  "symphonic black metal", "symphonic metal", "symphonic rock", "symphony", 
  "synth funk", "synth-pop", "synthwave", "t-pop", "tamil", "tamil soundtrack", 
  "tango", "tape music", "tech house", "tech trance", "technical death metal", 
  "techno", "teen pop", "tejano", "tex-mex", "texas blues", "texas country", 
  "third stream", "thrash metal", "timba", "traditional country", "traditional jazz", 
  "traditional pop", "trance", "trap", "trap edm", "trap latino", "trap metal", 
  "tribal ambient", "tribal house", "trip hop", "tropical house", "tropicália", 
  "turntablism", "twee pop", "uk drill", "uk garage", "underground hip hop", 
  "urban cowboy", "us power metal", "vallenato", "viking metal", "vocal", 
  "vocal jazz", "vocal trance", "vocalese", "waltz", "weltmusik", 
  "west coast hip hop", "western classical", "western swing", "wonky", "world", 
  "world fusion", "worship", "zamba", "zydeco"
];

    non_genre_tags = [
  "1–4 wochen", "2010s", "2014", "255", "5+ wochen", "acoustic", "animal on cover", 
  "annie moses band", "arrolladora", "arrolladora limon", "artist on cover", 
  "atmospheric", "award/qobuz/qobuzissime", "award/songlines/top of the world", "banda el limon", 
  "billboard hot 100", "brass", "butterfly on cover", "cam", 
  "carlos baute colgando en tus manos nada se compara a ti", "car on cover", 
  "cccd", "check it", "choral", "cover art recursion", "default", "divers", "duits", 
  "easy listening", "english", "ep", "experimental", "fusion", "gates", "genre", "gothic",
  "halloween", "horror", "indie", "instrumental", "interview", "island", 
  "kevin m. thomas", "la arrolladora", "la cama", "laut.de", "lil boosie", 
  "limon", "live", "los temerarios", "miscellaneous", "minecraft", "mod", "modern",
  "movie songs", "novelty", "offizielle charts", "parte 2", "podcast", "production",
  "progressive", "psychedelic", "regional", "remix", "self-titled", "somo", 
  "spoken word", "this glorious christmas", "todd in the shadows", "unknown", "unknown genre", 
  "vocal", "whale on cover", "write me back", "www.mzhiphop.com", "www.newsingles4u.com", 
  "year end chart", "jazzthing.de", "ballad", "non-music", "modern creative", "avant-garde", "berlin school",

    ];
    # Define a set of non-genre tags for faster lookup
    non_genre_tags_set = set(non_genre_tags)
    
    # Function to filter out non-genre tags from a genre string
    def filter_genres(genres_str: str) -> str:
        try:
            genres = eval(genres_str)  # Convert string representation of list to actual list
            return [genre for genre in genres if genre not in non_genre_tags_set]
        except:
            return []

    # Apply the filtering function to the 'genres' column
    df['genres'] = df['genres'].dropna().apply(filter_genres)
    
    # Save the cleaned dataset to a new CSV file
    df.to_csv("./billboard_album_final_super_genres.csv", index=False)


def year_stats():
    merged_df = pd.read_csv("./merged_dataset_billboard_download.csv")

    # Drop rows where release_date is NA
    merged_df = merged_df.dropna(subset=['release_date'])
    
    # Convert dates to years, handling any potential errors
    merged_df['release_year'] = pd.to_datetime(merged_df['release_date'], errors='coerce').dt.year
    
    # Drop any rows where the conversion failed
    merged_df = merged_df.dropna(subset=['release_year'])
    
    # Convert to integer after removing NA values
    merged_df['release_year'] = merged_df['release_year'].astype(int)
    
    release_year_counts = merged_df['release_year'].value_counts().sort_index(ascending=False)
    
    # Plotting the count of releases by year
    plt.figure(figsize=(12, 6))
    release_year_counts.plot(kind='line', color='skyblue')
    plt.title('Count of Releases by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('billboard_release_year_counts.png')
    plt.close()

    print(f"Total number of releases: {release_year_counts.sum()}")
    print(f"Percentage of total dataset: {(release_year_counts.sum()/len(merged_df))*100:.2f}%")
    print("\nDistribution by decade:")
    
    # Calculate distribution by decade
    decades = {}
    for year in release_year_counts.index:
        decade = (year // 10) * 10
        if decade in decades:
            decades[decade] += release_year_counts[year]
        else:
            decades[decade] = release_year_counts[year]
    
    for decade in sorted(decades.keys()):
        print(f"{decade}s: {decades[decade]} releases ({(decades[decade]/release_year_counts.sum())*100:.2f}%)")


def check_missing_genres_from11_genre_map():
    with open("../11_genre_map.txt", "r") as file:
        genre_map = json.load(file)

    df = pd.read_csv("./cleaned_billboard_dataset_no_non_genre_tags.csv")
    unique_genres = df['genres'].dropna().apply(eval).explode().unique()

    missing_genres = []
    for genre in unique_genres:
        if genre not in genre_map:
            missing_genres.append(genre)

    print(missing_genres)

def create_merged_dataset():
    # Create merged dataset
    merged_df = create_dataset(progress_folder)
    print(f"Total rows in merged dataset: {len(merged_df)}")
    merged_df.to_csv("./merged_dataset_billboard_download.csv", index=False)
    # Remove rows where 'found' is False
    merged_df = merged_df[merged_df['found_in_musicbrainz'] != False]
    
    # Save the cleaned dataset
    merged_df.to_csv("cleaned_merged_dataset.csv", index=False)

def remove_image_not_found():
    # Load the cleaned dataset
    df = pd.read_csv("../billboard_album_final_with_complexity.csv")

    # Get the list of MBIDs from the existing images
    existing_mbids = {filename.split('.')[0] for filename in os.listdir("../img/billboard")}

    # Filter the DataFrame to keep only rows where album_group_mbid is in the existing MBIDs
    df_filtered = df[df['album_group_mbid'].isin(existing_mbids)]

    # Save the filtered dataset
    df_filtered.to_csv("cleaned_merged_dataset_with_images.csv", index=False)

def duplicates_billboard():
    df = pd.read_csv('./cleaned_merged_dataset_with_images.csv')
    print(len(df))
    df = df.drop_duplicates(subset=['album_group_mbid'])
    # df2 = pd.read_csv('./results/billboard_album_with_complexity_cleaned.csv')
    # df = df.loc[~df['album_group_mbid'].isin(df2['album_group_mbid'])]
    print (len(df))
    # df.to_csv('./results/billboard_removed_from_cleaned.csv', index=False)


if __name__ == "__main__":
    # remove_duplicates()

    # progress_folder = "./progress"
    
    
    ##Create billboard dataset after musicbrainz download
    #create_merged_dataset()
    
    
    # # Create not found dataset
    # # not_found_df = create_not_found_dataset(merged_df, "./not_found_in_musicbrainz.csv")
    # # print(f"Total rows in not found dataset: {len(not_found_df)}")
    
    # # Get unique genres
    # unique_genres = get_unique_genres(merged_df)
    # print("\nUnique genres:")
    # # for genre in unique_genres:
    # #     print(f"- {genre}")
    # print(len(unique_genres))
    # # with open("unique_genres.txt", "w") as file:
    # #     for genre in unique_genres:
    # #         file.write(f"{genre}\n")
    
    # remove_non_genre_tags("./billboard_album_final_super_genres.csv")

    # year_stats()

    # check_missing_genres_from11_genre_map()
    # remove_image_not_found()
    duplicates_billboard()
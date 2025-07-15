import pandas as pd
from urllib.parse import urlparse

def is_valid_url(url):
    """Check if URL is valid and from a supported image hosting service."""
    if not isinstance(url, str):
        return False
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        
        # List of supported image hosting domains
        supported_domains = [
            'amazon.com',
            'amazonaws.com', 
            '7static.com',
            'discogs.com',
            'last.fm',
            'lastfm.freetls.fastly.net',
            'musicbrainz.org',
            'coverartarchive.org',
            'spotify.com',
            'scdn.co',
            'imgur.com',
            'wikimedia.org',
            'wikipedia.org'
        ]
        
        # Check if the domain matches any supported service
        domain = result.netloc.lower()
        return any(supported_domain in domain for supported_domain in supported_domains)
    except:
        return False

df = pd.read_csv('merged_dataset_renamed.csv')

# Check both URL columns
df['has_amazon_url'] = df['amazon_image_url'].apply(is_valid_url)
df['has_msdi_url'] = df['msdi_image_url'].apply(is_valid_url)
df['has_any_valid_url'] = df['has_amazon_url'] | df['has_msdi_url']

invalid_urls = df[~df['has_any_valid_url']]

print(f'Total albums in dataset: {len(df)}')
print(f'Albums with valid Amazon URLs: {df["has_amazon_url"].sum()}')
print(f'Albums with valid MSDI URLs: {df["has_msdi_url"].sum()}')
print(f'Albums with any valid URL: {df["has_any_valid_url"].sum()}')
print(f'Albums with no valid URLs: {len(invalid_urls)}')

print('\nFirst 5 albums with no valid URLs:')
for i, (_, row) in enumerate(invalid_urls.head(5).iterrows()):
    print(f"{i+1}. MBID: {row['album_group_mbid']}")
    print(f"   Album: {row['album_group_title']} by {row['artist_name']}")
    print(f"   Amazon URL: {row['amazon_image_url']}")
    print(f"   MSDI URL: {row['msdi_image_url']}")
    print()

# Show some examples of albums that now have valid URLs from MSDI
new_valid = df[~df['has_amazon_url'] & df['has_msdi_url']]
if len(new_valid) > 0:
    print(f'\nAlbums with valid MSDI URLs but no Amazon URL: {len(new_valid)}')
    print('First 5 examples:')
    for i, (_, row) in enumerate(new_valid.head(5).iterrows()):
        print(f"{i+1}. MBID: {row['album_group_mbid']}")
        print(f"   Album: {row['album_group_title']} by {row['artist_name']}")
        print(f"   MSDI URL: {row['msdi_image_url']}")
        print() 
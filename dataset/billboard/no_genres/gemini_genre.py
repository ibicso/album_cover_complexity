import pandas as pd
import google.generativeai as genai
import os

GEMINI_API_KEY = "AIzaSyBMC8_LdcQy5fuQGgV8v1mm2rKpOVzC4Rk"

# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)


def get_genre_from_ai(album_df_query):
    """
    Get music genres for albums using Google Gemini API
    
    Args:
        album_df_query: DataFrame or string containing album information
    
    Returns:
        str: Response from Gemini API with genre information
    """
    context = f"Please return the music genres for each album in the following dataframe, if you can't find the genre or you are not sure, return 'unknown'. Use this format genres: [genre1, genre2, ...]: {album_df_query}"
    
    try:
        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate content
        response = model.generate_content(context)
        
        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


def get_genres():
    """
    Process albums from CSV file and get their genres using AI
    """
    try:
        df = pd.read_csv('./billboard_album_no_genres.csv')
        
        # Process albums in batches to avoid hitting API limits
        batch_size = 200
        for i in range(2192, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Create a formatted string of the batch for the AI
            batch_info = ""
            for index, row in batch.iterrows():
                artist = row['artist']
                album = row['album_group_title']
                first_release_date = row['first_release_date_group']
                batch_info += f"Artist: {artist}, Album: {album}, Date: {first_release_date}\n"
            
            print(f"Processing batch {i//batch_size + 1}...")
            print(batch_info)
            
            # Get genres from AI
            genres_response = get_genre_from_ai(batch_info)
            if genres_response:
                print("AI Response:")
                print(genres_response)
                print("-" * 50)
                with open('album_genre_ai.txt', 'a') as f:
                    f.write(genres_response + '\n')
                    print(f"Written to file")
            else:
                print("Failed to get response for this batch")

                
    except FileNotFoundError:
        print("Error: Could not find the CSV file './billboard_album_no_genres.csv'")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


if __name__ == "__main__":
    get_genres()

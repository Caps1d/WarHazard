import pandas as pd
import numpy as np
import json
import spacy
from word2number import w2n
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from haversine import haversine


file_name = input('Enter the file name, json only: ')+'.json' 
dir = 'data/'
path = dir + file_name

with open(path) as f:
    data = json.load(f)

df = pd.DataFrame(data)

print('Data processing in progress, this will take around 2 minutes. Kick-back and relax!')

df["fatalities"] = pd.to_numeric(df["fatalities"], errors='coerce')

df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')

# Feature Engineering: 
# Extracting wounded data from the notes colulmn
nlp = spacy.load('en_core_web_md')

def extract_wounded(text):
    doc = nlp(text)
    wounded_count = 0

    for token in doc:
        if token.text.lower() == 'wounded':
            for child in token.children:
                if child.pos_ == 'NUM':
                    try:
                        wounded_count += int(child.text)
                    except ValueError:
                        wounded_count += w2n.word_to_num(child.text)
            if wounded_count == 0:
                for ancestor in token.ancestors:
                    if ancestor.pos_ == 'NUM':
                        try:
                            wounded_count += int(ancestor.text)
                        except ValueError:
                            wounded_count += w2n.word_to_num(ancestor.text)
                        break

    return wounded_count

df['wounded'] = df['notes'].apply(extract_wounded)

df["event_date"] = pd.to_datetime(df["event_date"])
df["month"] = df["event_date"].dt.month
df["year"] = df["event_date"].dt.year

# Combining fatalities + wounded into casualties
df['casualties'] = df['fatalities']+df['wounded']


grouped_columns = ["admin1", "admin2", "admin3", "location", "latitude", "longitude", "event_date", "year", "month", "event_type", "civilian_targeting"]

# Aggregate data by location, date, and event_type
df_agg = df.groupby(grouped_columns).agg(
    num_events=pd.NamedAgg(column='event_type', aggfunc='size'),
    total_casualties=pd.NamedAgg(column='casualties', aggfunc='sum')
).reset_index()

def haversine2(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in kilometers
    earth_radius = 6371

    # Calculate the distance
    distance = earth_radius * c
    return distance

#Define coordinates of the the battlefront towns
bakhmut_coords = (48.5956, 37.9999)
soledar_coords = (48.6833, 38.0667)
avdiivka_coords = (48.1394, 37.7497)
vuhledar_coords = (48.7798, 37.2490)
robotyne_coords = (47.44992394238662, 35.83787190517212)
kupiansk_coords = (49.7160738622855, 37.596104878691285)

#apply the haversine function to the dataset to calculate 
#the distances to each town and find the minimum distance:
df_agg['distance_to_bakhmut'] = haversine2(df_agg['latitude'], df_agg['longitude'], bakhmut_coords[0], bakhmut_coords[1])
df_agg['distance_to_soledar'] = haversine2(df_agg['latitude'], df_agg['longitude'], soledar_coords[0], soledar_coords[1])
df_agg['distance_to_avdiivka'] = haversine2(df_agg['latitude'], df_agg['longitude'], avdiivka_coords[0], avdiivka_coords[1])
df_agg['distance_to_vuhledar'] = haversine2(df_agg['latitude'], df_agg['longitude'], vuhledar_coords[0], vuhledar_coords[1])
df_agg['distance_to_robotyne'] = haversine2(df_agg['latitude'], df_agg['longitude'], robotyne_coords[0], robotyne_coords[1])
df_agg['distance_to_kupiansk'] = haversine2(df_agg['latitude'], df_agg['longitude'], kupiansk_coords[0], kupiansk_coords[1])


df_agg['min_distance_to_battlefront'] = df_agg[['distance_to_bakhmut', 'distance_to_soledar', 'distance_to_avdiivka', 'distance_to_vuhledar', 'distance_to_robotyne', 'distance_to_kupiansk']].min(axis=1)

# Drop the temporary distance columns
df_agg = df_agg.drop(columns=['distance_to_bakhmut', 'distance_to_soledar', 'distance_to_avdiivka', 'distance_to_vuhledar', 'distance_to_robotyne', 'distance_to_kupiansk'])

# One-hot Encoding categorical features
# Encode civilian_targeting as a binary column
df_agg['civilian_targeting_encoded'] = df_agg['civilian_targeting'].apply(lambda x: 1 if x == 'Civilian targeting' else 0)

# One-hot encode event_type and incorporate civilian_targeting_encoded for 'Explosions/Remote violence'
df_agg['event_battles'] = (df_agg['event_type'] == 'Battles').astype(int)
df_agg['event_explosions'] = ((df_agg['event_type'] == 'Explosions/Remote violence') & (df_agg['civilian_targeting_encoded'] == 0)).astype(int)
df_agg['event_explosions_civilians'] = ((df_agg['event_type'] == 'Explosions/Remote violence') & (df_agg['civilian_targeting_encoded'] == 1)).astype(int)
df_agg['event_violence_civilians'] = (df_agg['event_type'] == 'Violence against civilians').astype(int)

# t-SNE dimensionality reduction
# Selecting the columns we want to input into t-SNE
features = ['num_events', 'total_casualties', 'event_battles', 'event_explosions', 'event_explosions_civilians', 'event_violence_civilians']

# Normalize the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_agg[features])

# 2-D t-SNE embeddings 
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_features)
df_agg['tsne_0'] = tsne_results[:, 0]
df_agg['tsne_1'] = tsne_results[:, 1]

# Processing t-SNE data with KDE:
# KDE on t-SNE results
kde = gaussian_kde(np.vstack([df_agg['tsne_0'], df_agg['tsne_1']]))
density = kde(np.vstack([df_agg['tsne_0'], df_agg['tsne_1']]))

# Identify the densest point
densest_idx = np.argmax(density)
densest_point = (df_agg.iloc[densest_idx]['tsne_0'], df_agg.iloc[densest_idx]['tsne_1'])

# Compute distance to densest point
df_agg['distance_to_densest'] = np.sqrt((df_agg['tsne_0'] - densest_point[0])**2 + (df_agg['tsne_1'] - densest_point[1])**2)

# Convert into a distance score 
decay_factor = 0.05  # This is just an example value; adjust as needed
df_agg['tsne_distance_points'] = 50 * np.exp(-decay_factor * df_agg['distance_to_densest'])

# Compute physical distance score based on min_distance_to_battlefront:
df_agg['distance_points'] = 100 * np.exp(-decay_factor * df_agg['min_distance_to_battlefront'])

# Hazard score calculation:
# Combine tsne_distance_points with distance_points using the weights you provided to obtain the final hazard_score.
weight_distance = 0.7  # This is the weight for the distance to the battlefront
weight_tsne = 0.3     # This is the weight for the t-SNE derived score

df_agg['hazard_score'] = (df_agg['distance_points'] * weight_distance) + (df_agg['tsne_distance_points'] * weight_tsne)

# Normalize the hazard_score
min_hazard = df_agg['hazard_score'].min()
max_hazard = df_agg['hazard_score'].max()

# Apply Min-Max scaling to adjust scores between 0 and 100
df_agg['hazard_score'] = ((df_agg['hazard_score'] - min_hazard) / (max_hazard - min_hazard)) * 100
df_agg['hazard_score'] = df_agg['hazard_score'].round(0)

# Post-processing based on domain knowledge
df_agg.loc[df_agg['min_distance_to_battlefront'] <= 30, 'hazard_score'] = 100
df_agg.loc[df_agg['hazard_score'] < 30, 'hazard_score'] = 30
df_agg.loc[df_agg['event_type'] == 'Battles', 'hazard_score'] = 100

# Hazard score diffusion to nearby areas:

# Step 1: Extract high hazard locations
high_hazard_locs = df_agg[df_agg['hazard_score'] >= 80][['admin1','latitude', 'longitude', 'hazard_score']]
high_hazard_locs = high_hazard_locs.drop_duplicates(subset=['latitude', 'longitude'], keep=False)

# Step 2: Identify locations within 50km radius using row-wise function
def compute_distance(row, locations):
    if row['admin1'] in locations['admin1'].values:
        # find the index of the first occurrence
        idx = locations[locations['admin1']==row['admin1']].index[0]
        # vars for distance and score
        distance = 81 # this is a baseline value to flag the one's that are outside of radius
        hazard = 0 
        for i, r in locations.loc[idx:].iterrows():
            if r['admin1'] == row['admin1']:
                # compute distance between two locations 
                # check if its within 50km 
                delta = haversine((row['latitude'],row['longitude']), (r['latitude'],r['longitude']), unit='km')
                if delta <= distance:
                    distance = delta
                    hazard = r['hazard_score']
            else:
                break
        # apply exponential decay to distance
        hazard_decayed = 0.8*hazard * (1-0.02)**(distance)
        return row['hazard_score']+hazard_decayed
        
        
    return row['hazard_score']


df_agg['hazard_score'] = df_agg.apply(lambda x: compute_distance(x, high_hazard_locs) if x['hazard_score']<80 else x['hazard_score'], axis=1)

# Cliping and rounding values to be within our desired range
df_agg['hazard_score'] = df_agg['hazard_score'].clip(0,100)
df_agg['hazard_score'] = df_agg['hazard_score'].round(0)

# Formatting data to handle inconsistencies, those are just location names that have no influence on the scores
df_agg.loc[df_agg['admin2']=='Kyiv', 'admin3'] = 'Kyiv'
df_agg.loc[df_agg['location']=='Kherson', 'admin2'] = 'Khersonskyi'
df_agg = df_agg[~df_agg['admin3'].str.strip().eq('')]
df_agg.to_csv('Hazards_latest.csv', index=False)

# End of script print statement
print('Data stored as Hazards_latest.csv, remember to rename accordingly!')

#Batter perspective Analysis 

import datetime,requests
import http.client
import json
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
#from moviepy.editor import VideoFileClip
#import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation,PillowWriter
import streamlit as st
import imageio
import tempfile,os

def converter(gif_path):
    #os.popen("pip install imageio[ffmpeg]")
    #imageio.plugins.ffmpeg.download()
    try:
        with open(gif_path, 'rb') as f:
            st.write("Found GIF file:", gif_path)
    except FileNotFoundError:
        st.error("GIF file not found!")
        return

    try:
        # Create a temporary file to save the converted video
        with tempfile.NamedTemporaryFile(suffix=".mp4",prefix=gif_path[:-4]) as temp_file:
            gif_reader = imageio.get_reader(gif_path, format='GIF')
            fps = gif_reader.get_meta_data().get('fps', 1)  # Default to 1 FPS if metadata is missing

            #st.write(f"Temporary file created: {temp_file.name}")

            # Create a video writer for MP4 format
            with imageio.get_writer(temp_file.name, format='mp4', fps=fps) as video_writer:
                for frame in gif_reader:
                    video_writer.append_data(frame)

            st.success("Conversion successful!")
            st.video(temp_file.name)

            # Provide a download link for the converted MP4 file
            with open(temp_file.name, "rb") as f:
                st.download_button(label="Download MP4", data=f.read(), file_name=f"{gif_path[:-4]}.mp4", mime="video/mp4")
    except Exception as e:
        st.error(f"An error occurred during conversion: {e}")

def init():
    # Get today's date
    today = datetime.date.today() # Format the date as YYYY-MM-DD
    formatted_date = today.strftime("%Y-%m-%d")
    print(formatted_date)

    response = requests.get(
    f'https://www.sofascore.com/api/v1/sport/cricket/scheduled-events/{formatted_date}'
    )
    data=response.json()['events']
    diction={}
    for i in data:
        #if i['tournament']['uniqueTournament']['hasEventPlayerStatistics']==False:
            #continue
        print(i)
        print(i.keys())
        print(i['homeTeam']['name'])
        print(i['awayTeam']['name'])
        print(i['id'])
        #break
        try:
            #diction.update({f"{i['homeTeam']['name']} {i['homeScore']['display']} vs {i['awayTeam']['name']} {i['awayScore']['display']}":i['id']})
            diction.update({f"{i['homeTeam']['name']} vs {i['awayTeam']['name']}": i['id']})
        except KeyError:
            continue
            #diction.update({f"{i['homeTeam']['name']} vs {i['awayTeam']['name']}":i['id']})
    return diction
def get_matches(pid,matches=[], format="T20", ind=0):
  #if matches is None:
    #matches = []
  url = f"https://www.sofascore.com/api/v1/player/{pid}/events/last/{ind}"
  parsed = urlparse(url)
  conn = http.client.HTTPSConnection(parsed.netloc)
  conn.request("GET", parsed.path)
  res = conn.getresponse()
  data = res.read()
  #st.wite(data)
  jdata = json.loads(data.decode("utf-8"))
  #print(jdata['events'][0])
  try:
    for event in jdata['events']:
      ans=determine_match_format(event)
      print(ans)
      if format == ans:
        matches.append(event['id'])
        print(event['id'])
  except KeyError:
    return matches
  if jdata.get('hasNextPage'):
    get_matches(pid,matches, format, ind + 1)
  return matches
#init()
def determine_match_format(data):
    """Determines the cricket match format based on innings data.

    Returns:
        "T20", "ODI", "Test", or "Unknown format".
    """
    home_innings = data['homeScore'].get('innings', {})

    away_innings = data['awayScore'].get('innings', {})

    all_innings = list(home_innings.values()) + list(away_innings.values())

    print(all_innings,data['id'])
    if not all_innings:
        return "Unknown format"

    total_innings = len(all_innings)
    total_overs = 0

    for innings in all_innings:
        if 'overs' not in innings:  # Handle cases where 'overs' might be missing
            return "Unknown format"
        total_overs += innings['overs']

    # More robust logic based on total overs and number of innings
    if total_innings > 2: #most likely a test, but check overs
        if total_overs <= 200: #edge case for rain affected test match
            return "Test"
        else:
            return "Test"
    #elif total_innings == 4:
        #return "Test"
    elif total_innings == 2 and total_overs <= 40:
        return "T20"
    elif total_innings == 2 and total_overs <= 100:
       return "ODI"
    else:
        return "Unknown format"
def analyze_bowling_stats(det, bowling_type, player_slug):
    """
    Analyzes a bowler's stats against a specific batsman.

    Args:
        det: The dictionary containing bowling data.
        bowling_type: The key in the dictionary (e.g., 'Right-arm fast medium').
        player_slug: The slug of the bowler to analyze (e.g., 'ishant-sharma').

    Returns:
        A Pandas DataFrame with the bowler's stats, or None if no matching data is found.
    """

    if bowling_type not in det:
        print(f"Bowling type '{bowling_type}' not found in data.")
        return None

    bowling_data = det[bowling_type]
    bowlers = bowling_data['bowler']

    # Find indices where the bowler matches the player_slug
    matching_indices = [i for i, bowler in enumerate(bowlers) if bowler == player_slug]

    if not matching_indices:
        print(f"Bowler '{player_slug}' not found in '{bowling_type}' data.")
        return None

    # Extract relevant stats for the matching indices
    stats = {key: [value[i] for i in matching_indices] for key, value in bowling_data.items()}

    df = pd.DataFrame(stats)

    # Calculate additional stats
    df['dots'] = df['runs'].apply(lambda x: 1 if x == 0 or x== "W" else 0)
    df['is_boundary'] = df['runs'].apply(lambda x: 1 if x in [4, 6] else 0)
    df['balls']=df.shape[0]
    total_runs=df['runs'].apply(lambda x: 0 if x=='W' else x).sum()
    df['total_runs']=total_runs
    dots=df['dots'].sum()
    df['total_dots']=dots
    df['strike_rate']=(total_runs/df['balls'])*100
    boundaries=df['is_boundary'].sum()
    df['total_boundaries']=boundaries
    dot_percentage=(dots/df['balls'])*100
    boundary_percentage=(boundaries/df['balls'])*100
    df['dot_percentage']=dot_percentage
    df['boundary_percentage']=boundary_percentage
    zone_counts = df['zone'].value_counts().to_dict()
    for zone, count in zone_counts.items():
        df[f'runs_in_{zone}'] = df.loc[df['zone']==zone,'runs'].apply(lambda x: 0 if x=='W' else x).sum()
    df['wickets']=df['wicket'].apply(lambda x: 1 if x!='' else 0).sum()
    df.drop(['is_boundary','dots','runs','zone','x','y','length'])
    return df
def opp_team_venue(mid,pid):
    url = f"https://www.sofascore.com/api/v1/event/{mid}"
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(parsed.netloc)
    conn.request("GET", parsed.path)
    res = conn.getresponse()
    data = res.read()
    details = json.loads(data.decode("utf-8"))
    h_name=details['event']['homeTeam']['name']
    h_id=details['event']['homeTeam']['id']
    a_name=details['event']['awayTeam']['name']
    a_id=details['event']['awayTeam']['id']
    venue=details['event']['venue']['name']
    url = f"https://www.sofascore.com/api/v1/event/{mid}/lineups"
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(parsed.netloc)
    conn.request("GET", parsed.path)
    res = conn.getresponse()
    data = res.read()
    p_details = json.loads(data.decode("utf-8"))
    for team in ['home','away']:
        for player in p_details[team]['players']:
            if pid==player['player']['id']:
                if team == 'home':
                    return a_name,venue
                else:
                    return h_name,venue
#@st.cache_data
def append_bat_data(mid,pid,incidents=[]):
    #incidents=[]
    info = opp_team_venue(mid, pid)
    url = f"https://www.sofascore.com/api/v1/event/{mid}/incidents"
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(parsed.netloc)
    conn.request("GET", parsed.path)
    res = conn.getresponse()
    data = res.read()
    jdata = json.loads(data.decode("utf-8"))['incidents']
    for i in jdata:
        if i["batsman"]["id"] == pid:
            i['opp'] = info[0]
            i['venue'] = info[1]
            incidents.append(i)
    return incidents
def batter_ball_by_ball(incidents):
    det = {}
    for incident in incidents[::-1]:
        bowler_type = incident['bowler']['cricketPlayerInfo']['bowling']

        # Initialize bowler type entry if not already present
        if bowler_type not in det:
            det[bowler_type] = {
                "runs": [],
                "x": [],
                "y": [],
                "length": [],
                "angle": [],
                "bowler": [],
                "wicket": [],
                "zone": [],
                "opp":[],
                "venue":[]
            }
        # Debugging output
        # print("Bowl Detail:", j.get('bowlDetail'))
        # print("Bowler Slug:", j['bowler'].get('slug', 'No slug'))
        # Append ball data to respective lists
        if "wd" in incident.get("incidentClassLabel").lower():
            print("Wide!")
            continue
        # det[bowler_type]["runs"].append(incident['runs'])
        try:
            det[bowler_type]["x"].append(incident['ballDetails']['pitchHit']['x'])
            det[bowler_type]["y"].append(incident['ballDetails']['pitchHit']['y'])
            det[bowler_type]["opp"].append(incident['opp'])
            det[bowler_type]["venue"].append(incident['venue'])
            det[bowler_type]["length"].append(incident.get('length', 0))
            det[bowler_type]["zone"].append(incident.get('zone', ""))
            det[bowler_type]["angle"].append(incident.get('angle', 0))
            det[bowler_type]['bowler'].append(incident['bowler']['slug'])
            if incident.get('bowlDetail'):
                det[bowler_type]["wicket"].append(incident['bowlDetail'])
                det[bowler_type]["runs"].append("W")
            else:
                det[bowler_type]["wicket"].append("")
                det[bowler_type]["runs"].append(incident['runs'])
        except KeyError:
            continue
    return det
# Determine match format
#match_format = determine_match_format(data)
#print(f"The match format is: {match_format}")
def create_bat_animation(det,role):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    color_map = {0: 'gray', 1: 'blue', 2: 'yellow',3:'brown', 4: 'orange', 6: 'red',"W":'black'}
    circle = Circle((0, 0), radius=5, facecolor='green', edgecolor='black')
    ax1.add_patch(circle)

    # Create a Text object for the title *outside* the update function
    title_text = ax1.text(0.02, 1.05, "", transform=ax1.transAxes, fontsize=12, ha='left', va='top')
    stad_text = ax1.text(0.04, 1.15, "", transform=ax1.transAxes, fontsize=14, ha='left', va='top')
    bowlers=[]
    def update(frame):
        #ax1.clear()  # No longer needed to clear the whole axes
        #ax1.add_patch(circle) #no longer needed to add the circle again and again

        length = det[role]['length'][frame]
        angle = det[role]['angle'][frame]
        runs = det[role]['runs'][frame]
        bowler_name = det[role]['bowler'][frame]
        bowler_type = role

        x_end = length * np.cos(np.deg2rad(angle))
        y_end = length * np.sin(np.deg2rad(angle))
        color = color_map.get(runs)

        ax1.plot([0, x_end], [0, y_end], color=color, linewidth=2)
        venue_name = det[role]['venue'][frame]
        opp_name = det[role]['opp'][frame]
        stad_text.set_text(f"{venue_name} (vs {opp_name})")
        # Update the text of the title object
        title_text.set_text(f"{bowler_name} ({bowler_type})")
        try:
          if bowler_name != det[role]['bowler'][frame+1] and bowler_name not in bowlers:
              df =analyze_bowling_stats(det, role, det[role]['bowler'][frame])
              #st.session_state.df=mdf
              last_row = df.iloc[-1]
              print(f"{bowler_name} ({bowler_type})")
              #print(last_row)
              st.markdown(f"## {bowler_name} ({bowler_type})")
              st.dataframe(last_row.transpose())
              #visualize_batsman(st.session_state.df[1],st.session_state.df[0])
              bowlers.append(bowler_name)
        except:
          print("Last record")
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])

        for i, (runs, color) in enumerate(color_map.items()):
            ax1.text(-9.5, 8 - i * 0.7, f'Runs: {runs}', fontsize=12, va='top', color=color)

        # Scatter plot animation (unchanged)
        if frame < len(det[role]['runs']):
            runs_scatter = det[role]['runs'][:frame + 1]
            x_values = det[role]['x'][:frame + 1]
            y_values = det[role]['y'][:frame + 1]
            colors_scatter = [color_map.get(run) for run in runs_scatter]

            ax2.clear()
            ax2.scatter(x_values, y_values, color=colors_scatter)
            #ax2.set_xlim([0, max(det[role]['x']) + 5])
            #ax2.set_ylim([0, max(det[role]['y']) + 5])
            ax2.set_xlim([0, 50])
            ax2.set_ylim([0, 100])
            ax2.invert_yaxis()
            ax2.set_title("Wagon wheel (left) and Pitch plot (right)")

    ani = FuncAnimation(fig, update, frames=len(det[role]['runs']), repeat=False)
    gif_writer = PillowWriter(fps=1)
    ani.save(f'{st.session_state.player_name}_bat_animation_with_{role}.gif', writer=gif_writer)
    converter(f'{st.session_state.player_name}_bat_animation_with_{role}.gif')
    return f'{st.session_state.player_name}_bat_animation_with_{role}.mp4'

#Bowler perspective Analysis

#from defs import *
import datetime,requests
#from defs import get_matches
import http.client
import json
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.animation as animation
import streamlit as st
from matplotlib.animation import FuncAnimation,PillowWriter
import imageio
import tempfile
import os
#os.environ["PATH"] += os.pathsep + r'C:\ffmpeg-master-latest-win64-gpl\bin'
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
    st.write(a_id,a_name,h_name,h_id,venue)
    for team in ['home','away']:
        for player in p_details[team]['players']:
            if pid==player['player']['id']:
                if team == 'home':
                    st.write(a_name,venue)
                    return a_name,venue
                else:
                    st.write(h_name,venue)
                    return h_name,venue
    #return None
def get_matches(pid,matches=[], format="T20", ind=0):
  #if matches is None:
    #matches = []
  url = f"https://www.sofascore.com/api/v1/player/{pid}/events/last/{ind}"
  parsed = urlparse(url)
  conn = http.client.HTTPSConnection(parsed.netloc)
  conn.request("GET", parsed.path)
  res = conn.getresponse()
  data = res.read()
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
def analyze_batting_stats(det, role="Right", name="player1"):
    """
    Analyzes batting stats and returns the average stats per batting_type.

    Args:
        det: The dictionary containing batting data.

    Returns:
        A Pandas DataFrame with the average stats for each batting type.
    """

    all_stats = []

    for batting_type, batting_data in det.items():
        # Create DataFrame from batting_data
        df = pd.DataFrame(batting_data)

        # Convert 'runs' to numeric values, replacing 'W' with 0
        df['runs'] = df['runs'].apply(lambda x: 0 if x == 'W' else int(x))

        # Calculate additional stats
        df['dots'] = df['runs'].apply(lambda x: 1 if x == 0 else 0)
        df['is_boundary'] = df['runs'].apply(lambda x: 1 if x in [4, 6] else 0)
        df['balls'] = df.shape[0]
        total_runs = df['runs'].sum()
        df['total_runs'] = total_runs
        df['economy'] = df['total_runs'] / (df['balls'] / 6)
        total_dots = df['dots'].sum()
        df['total_dots'] = total_dots
        df['strike_rate'] = (total_runs / df['balls']) * 100
        total_boundaries = df['is_boundary'].sum()
        df['total_boundaries'] = total_boundaries
        dot_percentage = (total_dots / df['balls']) * 100
        boundary_percentage = (total_boundaries / df['balls']) * 100
        df['dot_percentage'] = dot_percentage
        df['boundary_percentage'] = boundary_percentage

        # Calculate total wickets
        df['wickets'] = df['wicket'].apply(lambda x: 1 if x != '' else 0)
        total_wickets = df['wickets'].sum()
        df['total_wickets'] = total_wickets

        # Calculate zone-specific stats
        zones = df['zone'].unique()
        for zone in zones:
            zone_df = df[df['zone'] == zone]
            wickets_in_zone = zone_df['wickets'].sum()
            runs_in_zone = zone_df['runs'].sum()
            df[f'wickets_in_{zone}'] = wickets_in_zone
            df[f'runs_in_{zone}'] = runs_in_zone

        # Select numeric columns for averaging
        numeric_cols = df.select_dtypes(include='number').columns

        # Compute average stats
        mean_stats = df[numeric_cols].mean()
        mean_stats['batting_type'] = batting_type  # Add batting type to the stats

        # Append the stats to the list
        all_stats.append(mean_stats)

    # Create a DataFrame with the average stats for each batting type
    avg_stats_df = pd.DataFrame(all_stats).reset_index(drop=True)
    # Find rows where 'batsman' equals batsman_name
    batsman_rows =df[df['batsman'] == name]

    # Find rows where 'batting_type' equals role
    role_rows =  avg_stats_df[ avg_stats_df['batting_type'] == role]

    return batsman_rows,role_rows

def create_ball_animation(det,role):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    color_map = {0: 'gray', 1: 'blue', 2: 'yellow',3:'brown', 4: 'orange', 6: 'red',"W":'black'}
    circle = Circle((0, 0), radius=5, facecolor='green', edgecolor='black')
    ax1.add_patch(circle)

    # Create a Text object for the title *outside* the update function
    title_text = ax1.text(0.02, 1.05, "", transform=ax1.transAxes, fontsize=12, ha='left', va='top')
    stad_text = ax1.text(0.04, 1.15, "", transform=ax1.transAxes, fontsize=14, ha='left', va='top')
    batters=[]
    def update(frame):
        #ax1.clear()  # No longer needed to clear the whole axes
        #ax1.add_patch(circle) #no longer needed to add the circle again and again

        length = det[role]['length'][frame]
        angle = det[role]['angle'][frame]
        runs = det[role]['runs'][frame]
        batsman_name = det[role]['batsman'][frame]
        batsman_type = role

        x_end = length * np.cos(np.deg2rad(angle))
        y_end = length * np.sin(np.deg2rad(angle))
        color = color_map.get(runs)

        ax1.plot([0, x_end], [0, y_end], color=color, linewidth=2)
        venue_name = det[role]['venue'][frame]
        opp_name = det[role]['opp'][frame]
        stad_text.set_text(f"{venue_name} (vs {opp_name})")
        # Update the text of the title object
        title_text.set_text(f"{batsman_name} ({batsman_type})")
        try:
          if batsman_name != det[role]['batsman'][frame+1] and batsman_name not in batters:
              #ax1.clear()
              df =analyze_batting_stats(det, role, det[role]['batsman'][frame])
              last_row = df.iloc[-1]
              print(f"{batsman_name} ({batsman_type})")
              print(last_row)
              st.markdown(f"## {batsman_name} ({batsman_type})")
              st.dataframe(last_row.transpose())
              batters.append(batsman_name)
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

    ani = animation.FuncAnimation(fig, update, frames=len(det[role]['runs']), repeat=False)
    #html = ani.to_jshtml()
    #st.components.v1.html(html, height=500)
    gif_writer = PillowWriter(fps=1)
    ani.save(f'{st.session_state.player_name}_bowl_animation_with_{role}.gif', writer=gif_writer)
    converter(f'{st.session_state.player_name}_bowl_animation_with_{role}.gif')
    return f'{st.session_state.player_name}_bowl_animation_with_{role}.mp4'
def bowler_ball_by_ball(incidents):
    det = {}
    for incident in incidents[::-1]:
        batter_type = incident["batsman"]['cricketPlayerInfo']['batting']

        # Initialize bowler type entry if not already present
        if batter_type not in det:
            det[batter_type] = {
                "runs": [],
                "x": [],
                "y": [],
                "length": [],
                "angle": [],
                "batsman": [],
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
        # det[batter_type]["runs"].append(incident['runs'])
        try:
            det[batter_type]["x"].append(incident['ballDetails']['pitchHit']['x'])
            det[batter_type]["y"].append(incident['ballDetails']['pitchHit']['y'])
            det[batter_type]["opp"].append(incident['opp'])
            det[batter_type]["venue"].append(incident['venue'])
            det[batter_type]["length"].append(incident.get('length', 0))
            det[batter_type]["zone"].append(incident.get('zone', ""))
            det[batter_type]["angle"].append(incident.get('angle', 0))
            det[batter_type]['batsman'].append(incident['batsman']['slug'])
            if incident.get('bowlDetail'):
                det[batter_type]["wicket"].append(incident['bowlDetail'])
                det[batter_type]["runs"].append("W")
            else:
                det[batter_type]["wicket"].append("")
                det[batter_type]["runs"].append(incident['runs'])
        except KeyError:
            continue
    return det
#@st.cache_data
def append_ball_data(mid,pid,incidents=[]):
    info = opp_team_venue(mid, pid)
    st.write(mid,pid)
    #incidents=[]
    url = f"https://www.sofascore.com/api/v1/event/{mid}/incidents"
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(parsed.netloc)
    conn.request("GET", parsed.path)
    res = conn.getresponse()
    data = res.read()
    jdata = json.loads(data.decode("utf-8"))['incidents']
    for i in jdata:
        if i["bowler"]["id"] == pid:
            i['opp'] = info[0]
            i['venue'] = info[1]
            incidents.append(i)
            #st.write(i)
    return incidents

def main(st):
    if st.session_state.switch==True:
        #recent = get_matches(786470, format="T20")[:10]
        incidents = []
        with st.spinner("Filtering data.."):
            for i in st.session_state.matches:
                try:
                    st.write(i, st.session_state.pid)
                    incidents = append_ball_data(i, st.session_state.pid, incidents)
                    #st.write(incidents)
                except KeyError:
                    continue
        st.session_state.incidents2 = incidents
        #st.write(st.session_state.incidents2)
        st.success("Filtering Success...")
        print(st.session_state)
    if st.session_state.incidents2:
        with st.spinner("Extracting ball by ball data"):
            det = bowler_ball_by_ball(st.session_state.incidents2)
        st.success("Extraction successful")
        st.session_state.det2 = det
    if st.session_state.det2:
        for role in st.session_state.det2:
            with st.spinner(f"Performance analysis vs {role}"):
                st.markdown(f"# vs {role}")
                st.write(det)
                create_ball_animation(det, role)
                #file = create_ball_animation(det, role)
                #video_file = open(file, 'rb')
                #video_bytes = video_file.read()
                #st.video(video_bytes)
                #video_file.close()
        st.success("Process Complete!!")

import streamlit as st
import datetime
from defs import *
import requests
from test import main
import os

st.title("Sofascore Tactical Analysis")
st.write(datetime.datetime.today())

# Define a button to start the analysis after choices are made
if 'match_selected' not in st.session_state:
    #os.popen("sudo apt update")
    #os.popen("pip install moviepy")
    #os.popen("ffmpeg -version")
    st.session_state.match_selected = False
    st.session_state.mid = None
    st.session_state.choose_side = None
    st.session_state.players = None
    st.session_state.pid = None
    st.session_state.player_name = None
    st.session_state.tid=None
    st.session_state.mformat= None
    st.session_state.recent_got=None
    st.session_state.matches=None
    st.session_state.incidents=None
    st.session_state.det=None
    st.session_state.incidents2 = None
    st.session_state.det2 = None
    st.session_state.switch=False
    st.session_state.nmat=None
    st.session_state.info=None
    st.session_state.ovr={}

def reset():
    st.session_state.match_selected = False
    st.session_state.mid = None
    st.session_state.choose_side = None
    st.session_state.players = None
    st.session_state.pid = None
    st.session_state.mformat = None
    st.session_state.recent_got = None
    st.session_state.matches = None
    st.session_state.incidents = None
    st.session_state.det = None
    st.session_state.incidents2 = None
    st.session_state.det2 = None
    st.session_state.switch = False
if st.button("Reset"):
    st.rerun()
if st.button("Start"):
    contents = init()  # Only calls match_id_init once
    choices = contents

    st.session_state.choices = choices
    st.session_state.match_selected = True

# Show dropdowns only after the "Start Analysis" button is clicked
if st.session_state.match_selected:
    choice = st.selectbox("Match", list(st.session_state.choices.keys()))
    if st.button("Selected"):
        match_id = st.session_state.choices[choice]
        st.write(f"Selected match: {choice}")
        st.write(f"Match ID: {match_id}")
        st.session_state.mid=match_id
if st.session_state.mid:
    side=st.radio("Home/Away",["homeTeam","awayTeam"])
    if st.button("Show Players"):
        st.session_state.choose_side=side
#st.write(st.session_state)
if st.session_state.choose_side and st.session_state.players is None:
    tid=requests.get(f"https://www.sofascore.com/api/v1/event/{st.session_state.mid}").json()['event'][st.session_state.choose_side]['id']
    st.session_state.tid=tid
    if st.session_state.tid:
        players={x["player"]['name']:x["player"]['id'] for x in requests.get(f"https://www.sofascore.com/api/v1/team/{tid}/players").json()['players']}
        st.session_state.players=players
#st.write(st.session_state.players)
if st.session_state.players is not None:
    player = st.selectbox("Players", list(st.session_state.players.keys()))
    st.session_state.player_name=player
if st.session_state.player_name is not None:
    pid = st.session_state.players[st.session_state.player_name]
    if st.button("Choose"):
        st.session_state.pid=pid
        st.write(pid,st.session_state.player_name)
if st.session_state.pid:
    mformat=st.selectbox("Choose format",["T20","ODI","Test"])
    if st.button("Select"):
        st.session_state.mformat=mformat
        #st.write(st.session_state)
if st.session_state.mformat and st.session_state.recent_got is None:
    with st.spinner("Getting recent data.."):
        recent=get_matches(st.session_state.pid,format=st.session_state.mformat)
    st.success("Got recent stats")
    st.session_state.recent_got=recent
if 'recent_got' in st.session_state and st.session_state.recent_got:
    #nmat=st.slider("Select number of matches",min_value=0,max_value=len(st.session_state.recent_got))
    st.success(f"Found data for {len(st.session_state.recent_got)} {st.session_state.mformat} matches")
    nmat=st.number_input(f"Enter an integer less than {len(st.session_state.recent_got)}")
    st.session_state.nmat = nmat
    st.write(st.session_state.nmat)
    if st.button("Done"):
        st.session_state.matches = st.session_state.recent_got[:int(st.session_state.nmat)]
    st.write(st.session_state.matches)
if st.session_state.matches:
    on=st.toggle("Keep it on to analyse batting..")
    if on:
        incidents=[]
        with st.spinner("Filtering data.."):
            for i in st.session_state.matches:
                try:
                    incidents=append_bat_data(i,st.session_state.pid,incidents)
                    st.session_state.incidents = incidents
                except KeyError as e:
                    st.error(e)
                    continue
        st.success("Filtering Success...")
    else:
        if st.button("Bowling"):
            st.session_state.switch=True
            main(st)

if st.session_state.incidents and st.session_state.switch ==False:
    with st.spinner("Extracting ball by ball data"):
        det=batter_ball_by_ball(st.session_state.incidents)
        st.session_state.det = det
    st.success("Extraction successful")
if st.session_state.det and st.session_state.switch ==False:
    for role in st.session_state.det:
        with st.spinner(f"Performance analysis vs {role}"):
            st.markdown(f"# vs {role}")
            create_bat_animation(det,role)
            #file=create_bat_animation(det,role)
            #video_file = open(file, 'rb')
            #video_bytes = video_file.read()
            #st.video(video_bytes)
            #video_file.close()
    st.success("Process Complete!!")

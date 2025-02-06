import pandas as pd
# icecream import ic
#pd.set_option('display.max_columns', None)
import pandas as pd
import streamlit as st
import pandas as pd
#bowling_pov
def visualize_bowler(role_stats,batsman_stats):
    #negatives=['economy','strike_rate','total_boundaries','boundary_percentage',"dot_percentage"]
    #st.write(role_stats,batsman_stats)
    #st.header("Performance Comparison")

    # Normalize role_stats keys by removing 'Average ' prefix
    normalized_role_stats = role_stats

    # Find common stats between batsman_stats and normalized_role_stats
    common_stats = set(batsman_stats.keys()).intersection(normalized_role_stats.keys())
    #ic(common_stats)
    # Initialize lists to store data for display
    stats_keys = []
    batsman_values = []
    role_average_values = []
    differences = []

    for stat in common_stats:
        batsman_value = list(batsman_stats[stat].values())[0]
        role_value = list(normalized_role_stats[stat].values())[0]

        # Check if both values are numeric
        if isinstance(batsman_value, (int, float)) and isinstance(role_value, (int, float)):
            difference = batsman_value - role_value
        else:
            # Skip non-numeric values or assign None
            difference = 0

        # Append the data for display
        stats_keys.append(stat)
        batsman_values.append(batsman_value)
        role_average_values.append(role_value)
        differences.append(difference)


    # Create columns for each stat
    num_stats = len(common_stats)
    cols = st.columns(num_stats)

    for a,row in enumerate(cols):
        #ic(stats_keys[a],
            #batsman_values[a],
            #differences[a])
        if batsman_values[a]==0 or stats_keys[a] in ['wickets','runs'] :
            continue
        if str(stats_keys[a]) in [ "dot" ,"balls","runs" ] :
            st.metric(
                label=stats_keys[a],
                value=batsman_values[a],
                delta=differences[a]
            )
        else:
            st.metric(
            label=stats_keys[a],
            value=batsman_values[a],
            delta=differences[a],
            delta_color='inverse'
            )
    return
@st.cache_data
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
        #ic(df['balls'])
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
    #st.write(role_rows)
    return batsman_rows.to_dict(),role_rows.to_dict()
#batting_pov
def visualize_batsman(role_stats,bowler_stats):
    # Normalize bowling_type_stats keys by removing 'Average ' prefix
    normalized_bowling_type_stats = role_stats

    # Find common stats between bowler_stats and normalized_bowling_type_stats
    common_stats = set(bowler_stats.keys()).intersection(normalized_bowling_type_stats.keys())

    # Initialize lists to store data for display
    stats_keys = []
    bowler_values = []
    bowling_type_values = []
    differences = []

    for stat in common_stats:
        bowler_value = bowler_stats[stat][0]
        role_value = normalized_bowling_type_stats[stat][0]

        # Check if both values are numeric
        if isinstance(bowler_value, (int, float)) and isinstance(role_value, (int, float)):
            difference = bowler_value - role_value
        else:
            # Skip non-numeric values or assign None
            difference = 0

        # Append the data for display
        stats_keys.append(stat)
        bowler_values.append(bowler_value)
        bowling_type_values.append(role_value)
        differences.append(difference)
        #ic(stat,bowler_value,role_value,difference)

    # Create columns for each stat
    num_stats = len(common_stats)
    cols = st.columns(num_stats)

    for idx, row in enumerate(cols):
        if bowler_values[idx] == 0 or stats_keys[idx] in ['wickets', 'runs']:
            continue

        if stats_keys[idx] not in ["dots", "balls", "runs"]:
            st.metric(
                label=stats_keys[idx],
                value=f"{bowler_values[idx]:.2f}",
                delta=f"{differences[idx]:+.2f}"
            )
        else:
            st.metric(
                label=stats_keys[idx],
                value=f"{bowler_values[idx]:.2f}",
                delta=f"{differences[idx]:+.2f}",
                delta_color='inverse'
            )
    return
@st.cache_data
def analyze_bowling_stats(det, bowler_name, bowling_type):

    """
    Calculates stats for a specific bowler.

    Args:
        det: The dictionary containing bowling data.
        bowler_name: The name of the bowler to analyze.
        bowling_type: The bowling type (e.g., 'Right-arm fast medium').

    Returns:
        A dictionary with the bowler's stats.
    """

    if bowling_type not in det:
        print(f"Bowling type '{bowling_type}' not found in data.")
        return None

    bowling_data = det[bowling_type]
    bowlers = bowling_data['bowler']

    # Find indices where the bowler matches bowler_name
    matching_indices = [i for i, bowler in enumerate(bowlers) if bowler == bowler_name]

    if not matching_indices:
        print(f"Bowler '{bowler_name}' not found in '{bowling_type}' data.")
        return None
    all_stats=[]
    # Extract relevant stats for the matching indices
    stats = {key: [value[i] for i in matching_indices] for key, value in bowling_data.items()}
    for batting_type, batting_data in det.items():
        df = pd.DataFrame(stats)

        # Convert 'runs' to numeric values, replacing 'W' with 0
        df['runs'] = df['runs'].apply(lambda x: 0 if x == 'W' else int(x))

        # Calculate additional stats
        df['dots'] = df['runs'].apply(lambda x: 1 if x == 0 else 0)
        df['is_boundary'] = df['runs'].apply(lambda x: 1 if x in [4, 6] else 0)
        #df['economy'] = df['balls'].apply(lambda x: (total_runs / x) * 6 if x > 0 else 0)
        df['balls'] = df.shape[0]
        #ic(df['balls'])
        total_runs = df['runs'].sum()
        #ic(total_runs)
        df['total_runs'] = total_runs
        #['economy'] = (total_runs / df['balls']) * 6 if df['balls'] > 0 else 0
        #df['economy'] = df['balls'].apply(lambda x: (total_runs / x) * 6 if x > 0 else 0)
        df['economy'] = df['total_runs'] / (df['balls'] / 6)
        dots = df['dots'].sum()
        df['total_dots'] = dots
        boundaries = df['is_boundary'].sum()
        df['total_boundaries'] = boundaries
        dot_percentage = (df['total_dots'] / df['balls']) * 100
        boundary_percentage = (df['total_boundaries'] / df['balls']) * 100
        df['dot_percentage'] = dot_percentage
        df['boundary_percentage'] = boundary_percentage
        df['wickets'] = df['wicket'].apply(lambda x: 1 if x != '' else 0)
        total_wickets = df['wickets'].sum()

        # Zone-specific stats
        zone_counts = df['zone'].value_counts().to_dict()
        for zone in zone_counts.keys():
            df[f'runs_in_{zone}'] = df.loc[df['zone'] == zone, 'runs'].sum()


        numeric_cols = df.select_dtypes(include='number').columns

            # Compute average stats
        mean_stats = df[numeric_cols].mean()
        mean_stats['bowler'] = bowler_name  # Add batting type to the stats
            # Append the stats to the list
        all_stats.append(mean_stats)
        # Create a DataFrame with the average stats for each batting type
    avg_stats_df = pd.DataFrame(all_stats).reset_index(drop=True)
        # Find rows where 'batsman' equals batsman_name
    bowler_rows = df[df['bowler'] == bowler_name]
    print(bowler_rows)
        # Find rows where 'batting_type' equals role
    role_rows = avg_stats_df[avg_stats_df['bowler'] == bowling_type]

    return bowler_rows.to_dict(),role_rows.to_dict()

if __name__ == "__main__":
    # Sample data dictionary
    sample_data = {
        "Right": {
            "batsman": ["player1", "player2", "player3", "player1", "player2"],
            "runs": [4, 6, 0, "W", 1],
            "zone": ["off", "on", "off", "on", "off"],
            "wicket": ["", "", "", "bowled", ""]
        },
        "Left": {
            "batsman": ["player1", "player2", "player3", "player4"],
            "runs": [1, 2, "W", 3],
            "zone": ["on", "off", "on", "off"],
            "wicket": ["", "", "caught", ""]
        }
    }

    # Test the function with sample data
    df = analyze_batting_stats(sample_data)
    visualize_bowler(df[1],df[0])

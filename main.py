import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import geopy.distance
from sklearn.neighbors import BallTree
from geopy.geocoders import ArcGIS
import traceback


@st.cache_data
def load_data():
    df = pd.read_csv('/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/data_final.csv')
    return df


def get_coordinates(location):
    geolocator = ArcGIS(user_agent="geoapiExercises")
    location = geolocator.geocode(location)
    return location.latitude, location.longitude


def get_locations_in_radius(df, lat, long, radius):
    # Convert latitude and longitude to radians
    df['lat_rad'], df['long_rad'] = np.radians(df['lat']), np.radians(df['lng'])
    loc_rad = np.array([np.radians(lat), np.radians(long)])

    # Convert radius from kilometers to radians
    radius = radius / 6371.0

    # Use BallTree to get locations within the given radius
    tree = BallTree(df[['lat_rad', 'long_rad']])
    distances, indices = tree.query([loc_rad], k=len(df))

    # Filter distances that are less than or equal to radius
    indices_within_radius = indices[0][distances[0] <= radius]

    # Filter dataframe to include only locations within the given radius
    df = df.loc[indices_within_radius]

    # Reset index of the dataframe
    df = df.reset_index(drop=True)

    return df


def calculate_zoom_level(df):
    latitudes = df['lat'].values
    longitudes = df['lng'].values

    min_lat, max_lat = np.min(latitudes), np.max(latitudes)
    min_long, max_long = np.min(longitudes), np.max(longitudes)

    distance_in_km = geopy.distance.geodesic((min_lat, min_long), (max_lat, max_long)).km

    # Adjust this factor as needed to tweak the zoom level
    # Higher values will zoom in further, lower values will zoom out
    zoom_factor = 0.5

    return max(0, 11 - np.log2(distance_in_km) * zoom_factor)


def main():
    df = load_data()

    st.title('Zip Code/City Information')

    location = st.text_input("Enter a City (eg Green Island, NY) or a Zip Code to begin")
    radius = st.slider("Enter Radius (in Kilometers)", min_value=1, max_value=100, value=50, step=1)

    if st.button("Search"):
        try:
            coords = get_coordinates(location)
            if coords is not None:
                lat, long = coords
                locations_in_radius_df = get_locations_in_radius(df.copy(), lat, long, radius)

                if not locations_in_radius_df.empty:
                    # Calculate the latitude and longitude range to set the initial map view
                    lat_range = [locations_in_radius_df['lat'].min(), locations_in_radius_df['lat'].max()]
                    lon_range = [locations_in_radius_df['lng'].min(), locations_in_radius_df['lng'].max()]

                    # Create a Pydeck map layer for the zip code locations
                    points_layer = pdk.Layer(
                        'ScatterplotLayer',
                        locations_in_radius_df,
                        get_position='[lng, lat]',
                        get_fill_color=[180, 0, 200, 255],
                        get_radius=1000,
                        pickable=True,
                        auto_highlight=True
                    )

                    # Create a Pydeck map layer for the search radius
                    circle_layer = pdk.Layer(
                        'CircleLayer',
                        pd.DataFrame({
                            'position': [[long, lat]],
                            'radius': [radius * 1000]  # The radius is expected to be in meters for CircleLayer
                        }),
                        get_position='position',
                        get_radius='radius',
                        get_fill_color=[255, 0, 0, 200],
                        pickable=True,
                    )

                    zoom_level = calculate_zoom_level(locations_in_radius_df)

                    # Combine the layers and create a Pydeck map
                    deck = pdk.Deck(
                        layers=[points_layer, circle_layer],
                        initial_view_state=pdk.ViewState(
                            latitude=np.mean(lat_range),
                            longitude=np.mean(lon_range),
                            zoom=zoom_level,
                            pitch=0,
                        ),
                        tooltip={
                            'html': '<b>Zip:</b> {zip}<br/><b>City:</b> {city}',
                            'style': {
                                'backgroundColor': 'steelblue',
                                'color': 'white'
                            }
                        },
                        map_style="mapbox://styles/mapbox/light-v9"
                    )

                    st.pydeck_chart(deck)

                    # Perform necessary calculations and display results.

                    # Calculate total values for certain columns
                    total_population = locations_in_radius_df['population'].sum()
                    total_households = locations_in_radius_df['households'].sum()
                    total_labor_force = locations_in_radius_df['labor_force'].sum()

                    # Calculate weighted averages
                    weighted_avg_age = np.average(locations_in_radius_df['age_median'],
                                                  weights=locations_in_radius_df['population'])
                    weighted_avg_family_size = np.average(locations_in_radius_df['family_size'],
                                                          weights=locations_in_radius_df['population'])
                    weighted_avg_income = np.average(locations_in_radius_df['income_household_median'],
                                                     weights=locations_in_radius_df['households'])

                    # Calculate ratios
                    male_ratio = locations_in_radius_df['male'].sum() / total_population
                    female_ratio = locations_in_radius_df['female'].sum() / total_population
                    married_ratio = locations_in_radius_df['married'].sum() / total_population
                    education_college_or_above_ratio = locations_in_radius_df[
                                                           'education_college_or_above'].sum() / total_population

                    race_white_ratio = locations_in_radius_df['race_white'].sum() / total_population
                    race_black_ratio = locations_in_radius_df['race_black'].sum() / total_population
                    race_asian_ratio = locations_in_radius_df['race_asian'].sum() / total_population
                    race_native_ratio = locations_in_radius_df['race_native'].sum() / total_population
                    race_pacific_ratio = locations_in_radius_df['race_pacific'].sum() / total_population
                    race_other_ratio = locations_in_radius_df['race_other'].sum() / total_population
                    race_multiple_ratio = locations_in_radius_df['race_multiple'].sum() / total_population

                    unemployment_ratio = locations_in_radius_df['unemployed'].sum() / total_labor_force

                    home_ownership_ratio = locations_in_radius_df['home_own'].sum() / total_households
                    households_six_figures_or_above_ratio = locations_in_radius_df[
                                                                'households_six_fig'].sum() / total_households

                    # Display results
                    st.markdown(f"**Total Population:** {total_population:,}")
                    st.markdown(f"**Median Age:** {weighted_avg_age:.1f}")
                    st.markdown(f"**Average Family Size:** {weighted_avg_family_size:.1f}")
                    st.markdown(f"**Median Household Income:** ${weighted_avg_income:,.0f}")

                    st.markdown(f"**% Male:** {male_ratio * 100:.2f}%")
                    st.markdown(f"**% Female:** {female_ratio * 100:.2f}%")
                    st.markdown(f"**% Married:** {married_ratio * 100:.2f}%")
                    st.markdown(
                        f"**% With College Degrees:** {education_college_or_above_ratio * 100:.2f}%")

                    st.markdown(f"**White:** {race_white_ratio * 100:.2f}%")
                    st.markdown(f"**Black:** {race_black_ratio * 100:.2f}%")
                    st.markdown(f"**Asian:** {race_asian_ratio * 100:.2f}%")
                    st.markdown(f"**Native American:** {race_native_ratio * 100:.2f}%")
                    st.markdown(f"**Pacific Islander:** {race_pacific_ratio * 100:.2f}%")
                    st.markdown(f"**Multiple Races:** {race_multiple_ratio * 100:.2f}%")
                    st.markdown(f"**Race (Other):** {race_other_ratio * 100:.2f}%")

                    st.markdown(f"**Unemployment Rate:** {unemployment_ratio * 100:.2f}%")

                    st.markdown(f"**Home Ownership Rate:** {home_ownership_ratio * 100:.2f}%")
                    st.markdown(
                        f"**% of Households Earning 100,000 or More:**"
                        f" {households_six_figures_or_above_ratio * 100:.2f}%")

                else:
                    st.error("No locations found within the provided radius.")

            else:
                st.error("Failed to get coordinates for the provided location.")

        except Exception as e:
            st.error(f"Error when trying to search: {e}")
            st.text(traceback.format_exc())  # This will print the full traceback


if __name__ == '__main__':
    main()

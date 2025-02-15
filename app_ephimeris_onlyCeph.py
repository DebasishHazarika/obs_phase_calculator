import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from astropy.time import Time
import io

# Set page config for a wider layout
st.set_page_config(layout="wide")

def jd_to_phase(t, period, t0):
    """Convert Julian Date to phase using period and t0"""
    phase = ((t - t0) / period) % 1
    return phase

def create_fourier_model(phase, period, amplitude, r21, phi21, r31, phi31):
    """Create Fourier model for the light curve"""
    omega = 2 * np.pi
    # First term
    c1 = amplitude / (1 + r21 + r31)
    # Calculate other coefficients
    c2 = r21 * c1
    c3 = r31 * c1
    # Calculate phases
    phi1 = 0  # Reference phase
    phi2 = phi21 + 2 * phi1
    phi3 = phi31 + 3 * phi1
    
    # Construct the model
    model = (c1 * np.cos(omega * (phase - 0) + phi1) +
            c2 * np.cos(2 * omega * (phase - 0) + phi2) +
            c3 * np.cos(3 * omega * (phase - 0) + phi3))
    return model

def utc_to_jd(date_str, time_str):
    """Convert UTC date and time string to JD-2450000"""
    datetime_str = f"{date_str} {time_str}"
    t = Time(datetime_str, format='iso', scale='utc')
    return t.jd - 2450000

def load_and_clean_data(file):
    """Load data from txt file and clean it"""
    # Skip the first 6 lines and use the 7th line as header
    df = pd.read_csv(file, skiprows=6, delimiter='\t')
    
    # Remove rows with missing data or -99.99
    df = df[df['R31_1'] != -99.99]  # Filter out rows with -99.99
    df = df.dropna()  # Remove rows with any missing values
    
    return df

# Title and file upload
st.title('Cepheid Variable Analysis Tool')

# File upload
uploaded_file = st.file_uploader("Upload Cepheid catalog (TXT format)", type=['txt'])

if uploaded_file is not None:
    # Read and clean the catalog
    df = load_and_clean_data(uploaded_file)
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Sky region parameters
        st.subheader("Sky Region Parameters")
        ra_center = st.number_input("RA Center (deg)", min_value=0.0, max_value=360.0, value=80.0)
        dec_center = st.number_input("Dec Center (deg)", min_value=-90.0, max_value=90.0, value=-69.0)
        width = st.number_input("Width (deg)", min_value=0.1, max_value=10.0, value=1.0)
        height = st.number_input("Height (deg)", min_value=0.1, max_value=10.0, value=1.0)
    
    with col2:
        # Observation time parameters
        st.subheader("Observation Time")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        
        # Add time selection
        start_time = st.time_input("Start Time", value=time(20, 0))  # Default 20:00
        end_time = st.time_input("End Time", value=time(23, 59))    # Default 23:59
        
        # Convert dates and times to JD
        start_jd = utc_to_jd(start_date.strftime('%Y-%m-%d'), 
                            start_time.strftime('%H:%M:%S'))
        end_jd = utc_to_jd(end_date.strftime('%Y-%m-%d'), 
                          end_time.strftime('%H:%M:%S'))
    
    # Filter Cepheids in the selected region
    mask = ((df['RA'] >= ra_center - width/2) & 
            (df['RA'] <= ra_center + width/2) & 
            (df['Decl'] >= dec_center - height/2) & 
            (df['Decl'] <= dec_center + height/2))
    selected_cepheids = df[mask].copy()
    
    # Display number of selected Cepheids
    st.write(f"Number of Cepheids in selected region: {len(selected_cepheids)}")
    
    # Create plots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sky plot
    ax1.scatter(df['RA'], df['Decl'], color='gray', alpha=0.3, s=10, label='All Cepheids')
    ax1.scatter(selected_cepheids['RA'], selected_cepheids['Decl'], color='red', s=50, label='Selected')
    ax1.set_xlabel('RA (deg)')
    ax1.set_ylabel('Dec (deg)')
    ax1.set_title('Sky Region')
    ax1.legend()
    
    # Light curves (using V magnitude)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_cepheids)))
    phase_array = np.linspace(0, 1, 100)
    
    for idx, (_, cepheid) in enumerate(selected_cepheids.iterrows()):
        # Generate model light curve
        mag_model = create_fourier_model(phase_array, cepheid['P_1'], cepheid['A_1'],
                                       cepheid['R21_1'], cepheid['phi21_1'],
                                       cepheid['R31_1'], cepheid['phi31_1'])
        
        # Plot model (add V magnitude offset)
        ax2.plot(phase_array, mag_model + cepheid['V'], 
                color=colors[idx % len(colors)], alpha=0.5,
                label=f"OGLE-LMC-CEP-{idx+1:04d}")
        
        # Calculate and plot observation phases
        obs_phases = jd_to_phase(np.array([start_jd, end_jd]), 
                               cepheid['P_1'], 
                               cepheid['T0_1'])
        model_at_obs = create_fourier_model(obs_phases, cepheid['P_1'], cepheid['A_1'],
                                          cepheid['R21_1'], cepheid['phi21_1'],
                                          cepheid['R31_1'], cepheid['phi31_1'])
        ax2.scatter(obs_phases, model_at_obs + cepheid['V'],
                   color=colors[idx % len(colors)], s=100, marker='*')
    
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('V Magnitude')
    ax2.set_title('Light Curves')
    ax2.invert_yaxis()  # Astronomical convention
    if len(selected_cepheids) <= 10:  # Only show legend if there are 10 or fewer Cepheids
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig1)
    
    # Create observation summary dataframe
    obs_summary = []
    for _, cepheid in selected_cepheids.iterrows():
        start_phase = jd_to_phase(start_jd, cepheid['P_1'], cepheid['T0_1'])
        end_phase = jd_to_phase(end_jd, cepheid['P_1'], cepheid['T0_1'])
        obs_summary.append({
            'OGLE_ID': cepheid.name,  # Using the index as ID
            'RA': cepheid['RA'],
            'Decl': cepheid['Decl'],
            'V_mag': cepheid['V'],
            'Period': cepheid['P_1'],
            'Start_JD': f"{start_jd:.5f}",
            'End_JD': f"{end_jd:.5f}",
            'Start_Phase': f"{start_phase:.3f}",
            'End_Phase': f"{end_phase:.3f}",
            'Start_UTC': f"{start_date} {start_time}",
            'End_UTC': f"{end_date} {end_time}"
        })
    
    obs_df = pd.DataFrame(obs_summary)
    
    # Display and download options
    st.subheader("Observation Summary")
    st.dataframe(obs_df)
    
    # Download buttons
    col3, col4 = st.columns(2)
    
    with col3:
        # Download plot
        buf = io.BytesIO()
        fig1.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        btn = st.download_button(
            label="Download Plot",
            data=buf.getvalue(),
            file_name="cepheid_plot.png",
            mime="image/png"
        )
    
    with col4:
        # Download data
        csv = obs_df.to_csv(index=False)
        btn = st.download_button(
            label="Download Data",
            data=csv,
            file_name="observation_summary.csv",
            mime="text/csv"
        )
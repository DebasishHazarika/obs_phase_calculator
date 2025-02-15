# observational planner for Cepheids 
# based on MC Cepheids and RRLyr from OGLE IV : https://ogledb.astrouw.edu.pl/~ogle/OCVS/
# v1 - debasish, Feb 15, 2025
#
# check: need to add asynchronize call to reduce loading times
#

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
    df = df[(df['R31_1'] != -99.99) & (df['V'] != -99.99)]  # Filter out rows with -99.99
    df = df.dropna()  # Remove rows with any missing values
    
    return df


# Streamlit UI Components
st.markdown('<h3 style="font-size: 32px; color: #2ecc71;">Variable Star Obs Phase Calculator</h3>', unsafe_allow_html=True)
st.markdown('<h5 style="font-size: 18px; color: #3498db;">#v1. Debasish Feb 15, 2025 // debasish.academic@gmail.com</h5>', unsafe_allow_html=True)

# File upload for both types of stars
col1, col2 = st.columns(2)
with col1:
    cep_file = st.file_uploader("Upload Cepheid catalog (TXT format)", type=['txt'])
with col2:
    rrl_file = st.file_uploader("Upload RR Lyrae catalog (TXT format)", type=['txt'])

if cep_file is not None or rrl_file is not None:
    # Load available data
    cep_df = load_and_clean_data(cep_file) if cep_file else None
    rrl_df = load_and_clean_data(rrl_file) if rrl_file else None
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Sky region parameters
        st.subheader("Sky Region to Observe")
        ra_center = st.number_input("RA Center (deg)", min_value=0.0, max_value=360.0, value=80.0)
        dec_center = st.number_input("Dec Center (deg)", min_value=-90.0, max_value=90.0, value=-69.0)
        width = st.number_input("Width (deg)", min_value=0.1, max_value=10.0, value=1.0)
        height = st.number_input("Height (deg)", min_value=0.1, max_value=10.0, value=1.0)
    
    with col2:
        # Observation time parameters
        st.subheader("Observation Time")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        start_time = st.time_input("Start Time", value=time(20, 0))
        end_time = st.time_input("End Time", value=time(23, 59))
        
        # Convert dates and times to JD
        start_jd = utc_to_jd(start_date.strftime('%Y-%m-%d'), 
                            start_time.strftime('%H:%M:%S'))
        end_jd = utc_to_jd(end_date.strftime('%Y-%m-%d'), 
                          end_time.strftime('%H:%M:%S'))


    # Create figure with four subplots
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])
    ax_sky_cep = fig.add_subplot(gs[0:2, 0])  # Cepheid sky plot
    ax_sky_rrl = fig.add_subplot(gs[2:4, 0])  # RR Lyrae sky plot
    ax_cep = fig.add_subplot(gs[0:2, 1])      # Cepheid light curves
    ax_rrl = fig.add_subplot(gs[2:4, 1])      # RR Lyrae light curves
    
    # Cepheid sky plot
    if cep_df is not None:
        mask_cep = ((cep_df['RA'] >= ra_center - width/2) & 
                    (cep_df['RA'] <= ra_center + width/2) & 
                    (cep_df['Decl'] >= dec_center - height/2) & 
                    (cep_df['Decl'] <= dec_center + height/2))
        selected_cep = cep_df[mask_cep].copy()
        ax_sky_cep.scatter(cep_df['RA'], cep_df['Decl'], color='gray', alpha=0.3, s=10, label='All OGLE')
        ax_sky_cep.scatter(selected_cep['RA'], selected_cep['Decl'], color='red', s=10, label='Selected')
        
        # Add grid and region box for Cepheids
        ax_sky_cep.grid(alpha=0.02)
        rect = plt.Rectangle((ra_center - width/2, dec_center - height/2), 
                            width, height, fill=False, color='blue', linestyle='-', lw=2)
        ax_sky_cep.add_patch(rect)
        
        # Set sky plot limits and labels for Cepheids
        ax_sky_cep.set_xlim(65, 100)
        ax_sky_cep.set_ylim(-76, -60)
        ax_sky_cep.set_xlabel('RA (deg)', fontsize=12)
        ax_sky_cep.set_ylabel('Dec (deg)', fontsize=12)
        ax_sky_cep.set_title('Cepheid Sky Region')
        ax_sky_cep.legend(loc='upper center', fontsize=10)

        # Add selection info to Cepheid sky plot
        info_text = f"Selected Cepheids: {len(selected_cep)}\n"
        if len(selected_cep) > 0:
            info_text += f"P range: {selected_cep['P_1'].min():.3f} - {selected_cep['P_1'].max():.3f} days"
        ax_sky_cep.text(0.02, 0.98, info_text, transform=ax_sky_cep.transAxes, 
                       verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add period histogram inset for Cepheids
        if len(selected_cep) > 0:
            inset_cep = ax_sky_cep.inset_axes([0.65, 0.65, 0.34, 0.34])
            inset_cep.hist(selected_cep['P_1'], bins='auto', color='red', alpha=0.7)
            inset_cep.set_xlabel('Period (days)')
            inset_cep.set_ylabel('N')

    # RR Lyrae sky plot
    if rrl_df is not None:
        mask_rrl = ((rrl_df['RA'] >= ra_center - width/2) & 
                    (rrl_df['RA'] <= ra_center + width/2) & 
                    (rrl_df['Decl'] >= dec_center - height/2) & 
                    (rrl_df['Decl'] <= dec_center + height/2))
        selected_rrl = rrl_df[mask_rrl].copy()
        ax_sky_rrl.scatter(rrl_df['RA'], rrl_df['Decl'], color='lightgray', alpha=0.3, s=10, label='All OGLE')
        ax_sky_rrl.scatter(selected_rrl['RA'], selected_rrl['Decl'], color='blue', s=5, label='Selected')

        # Add grid and region box for RR Lyrae
        ax_sky_rrl.grid(alpha=0.02)
        rect = plt.Rectangle((ra_center - width/2, dec_center - height/2), 
                           width, height, fill=False, color='red', linestyle='-')
        ax_sky_rrl.add_patch(rect)
        
        # Set sky plot limits and labels for RR Lyrae
        ax_sky_rrl.set_xlim(60, 110)
        ax_sky_rrl.set_ylim(-85, -55)
        ax_sky_rrl.set_xlabel('RA (deg)', fontsize=12)
        ax_sky_rrl.set_ylabel('Dec (deg)', fontsize=12)
        ax_sky_rrl.set_title('RR Lyrae Sky Region')
        ax_sky_rrl.legend(loc='upper center', fontsize=10)

        # Add selection info to RR Lyrae sky plot
        info_text = f"Selected RR Lyrae: {len(selected_rrl)}\n"
        if len(selected_rrl) > 0:
            info_text += f"P range: {selected_rrl['P_1'].min():.3f} - {selected_rrl['P_1'].max():.3f} days"
        ax_sky_rrl.text(0.02, 0.98, info_text, transform=ax_sky_rrl.transAxes, 
                       verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add period histogram inset for RR Lyrae
        if len(selected_rrl) > 0:
            inset_rrl = ax_sky_rrl.inset_axes([0.65, 0.65, 0.34, 0.34])
            inset_rrl.hist(selected_rrl['P_1'], bins='auto', color='blue', alpha=0.7)
            inset_rrl.set_xlabel('Period (days)')
            inset_rrl.set_ylabel('N')


    # Light curves
    phase_array = np.linspace(0, 1, 100)
    
    # Cepheid light curves
    if cep_df is not None and len(selected_cep) > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_cep)))
        for idx, (_, cepheid) in enumerate(selected_cep.iterrows()):
            mag_model = create_fourier_model(phase_array, cepheid['P_1'], cepheid['A_1'],
                                           cepheid['R21_1'], cepheid['phi21_1'],
                                           cepheid['R31_1'], cepheid['phi31_1'])
            ax_cep.plot(phase_array, mag_model + cepheid['V'], 
                    color=colors[idx % len(colors)], alpha=0.5,
                    label=f"OGLE-LMC-CEP-{idx+1:04d}")
            
            obs_phases = jd_to_phase(np.array([start_jd, end_jd]), 
                                   cepheid['P_1'], 
                                   cepheid['T0_1'])
            model_at_obs = create_fourier_model(obs_phases, cepheid['P_1'], cepheid['A_1'],
                                              cepheid['R21_1'], cepheid['phi21_1'],
                                              cepheid['R31_1'], cepheid['phi31_1'])
            ax_cep.scatter(obs_phases, model_at_obs + cepheid['V'],
                       color=colors[idx % len(colors)], s=100, marker='*')

    # Create the observation times string
    obs_times_text = f"Obs times: {start_date}//{start_time} to {end_date}//{end_time}"

    ax_cep.set_xlabel('Phase')
    ax_cep.set_ylabel('V Mag (OGLE IV)')
    ax_cep.set_title(f'Cepheid {obs_times_text}')
    ax_cep.invert_yaxis()
    if cep_df is not None and len(selected_cep) <= 10:
        ax_cep.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # RR Lyrae light curves
    if rrl_df is not None and len(selected_rrl) > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_rrl)))
        for idx, (_, rrl) in enumerate(selected_rrl.iterrows()):
            mag_model = create_fourier_model(phase_array, rrl['P_1'], rrl['A_1'],
                                           rrl['R21_1'], rrl['phi21_1'],
                                           rrl['R31_1'], rrl['phi31_1'])
            ax_rrl.plot(phase_array, mag_model + rrl['V'], 
                    color=colors[idx % len(colors)], alpha=0.5,
                    label=f"OGLE-LMC-RRLYR-{idx+1:04d}")
            
            obs_phases = jd_to_phase(np.array([start_jd, end_jd]), 
                                   rrl['P_1'], 
                                   rrl['T0_1'])
            model_at_obs = create_fourier_model(obs_phases, rrl['P_1'], rrl['A_1'],
                                              rrl['R21_1'], rrl['phi21_1'],
                                              rrl['R31_1'], rrl['phi31_1'])
            ax_rrl.scatter(obs_phases, model_at_obs + rrl['V'],
                       color=colors[idx % len(colors)], s=100, marker='*')

    ax_rrl.set_xlabel('Phase', fontsize=12)
    ax_rrl.set_ylabel('V Mag (OGLE IV)', fontsize=12)
    ax_rrl.set_title(f'RRLyr {obs_times_text}')
    ax_rrl.invert_yaxis()
    if rrl_df is not None and len(selected_rrl) <= 10:
        ax_rrl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Create observation summary dataframes
    if cep_df is not None and len(selected_cep) > 0:
        st.subheader("Cepheid Observation Summary")
        cep_summary = []
        for _, cepheid in selected_cep.iterrows():
            start_phase = jd_to_phase(start_jd, cepheid['P_1'], cepheid['T0_1'])
            end_phase = jd_to_phase(end_jd, cepheid['P_1'], cepheid['T0_1'])
            cep_summary.append({
                'OGLE_ID': cepheid.name,
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
        cep_df_summary = pd.DataFrame(cep_summary)
        st.dataframe(cep_df_summary)
        
        # Download Cepheid data
        csv_cep = cep_df_summary.to_csv(index=False)
        st.download_button(
            label="Download Cepheid Data",
            data=csv_cep,
            file_name="cepheid_observation_summary.csv",
            mime="text/csv"
        )
    
    if rrl_df is not None and len(selected_rrl) > 0:
        st.subheader("RR Lyrae Observation Summary")
        rrl_summary = []
        for _, rrl in selected_rrl.iterrows():
            start_phase = jd_to_phase(start_jd, rrl['P_1'], rrl['T0_1'])
            end_phase = jd_to_phase(end_jd, rrl['P_1'], rrl['T0_1'])
            rrl_summary.append({
                'OGLE_ID': rrl.name,
                'RA': rrl['RA'],
                'Decl': rrl['Decl'],
                'V_mag': rrl['V'],
                'Period': rrl['P_1'],
                'Start_JD': f"{start_jd:.5f}",
                'End_JD': f"{end_jd:.5f}",
                'Start_Phase': f"{start_phase:.3f}",
                'End_Phase': f"{end_phase:.3f}",
                'Start_UTC': f"{start_date} {start_time}",
                'End_UTC': f"{end_date} {end_time}"
            })
        rrl_df_summary = pd.DataFrame(rrl_summary)
        st.dataframe(rrl_df_summary)

        # Download RR Lyrae data
        csv_rrl = rrl_df_summary.to_csv(index=False)
        st.download_button(
            label="Download RR Lyrae Data",
            data=csv_rrl,
            file_name="rrlyr_observation_summary.csv",
            mime="text/csv"
        )
    
    # Download plot button
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    st.download_button(
        label="Download Plot",
        data=buf.getvalue(),
        file_name="variable_stars_plot.png",
        mime="image/png"
    )

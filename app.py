"""
Streamlit Web Application for TNT Tokenomics Simulation

This application provides an interactive interface for exploring tokenomics scenarios
using the core simulation engine. Users can adjust parameters and visualize the
impact on circulating supply, net flows, and other key metrics using Altair charts.
"""

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

# Import core simulation components
# Note: Ensure the core simulation file is in the same directory or Python path
from sim import (
    TokenomicsSimulation, 
    SimulationConfig, 
    ConstantPrice, 
    LinearPrice, 
    ExponentialPrice
)

# Configure Streamlit page
st.set_page_config(
    page_title="TNT Tokenomics Simulation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_config() -> tuple[SimulationConfig, object]:
    """
    Create sidebar configuration interface with organized parameter groups
    
    Returns:
        Tuple of (SimulationConfig, PriceTrajectory)
    """
    st.sidebar.title("Simulation Configuration")
    st.sidebar.markdown("Adjust parameters to explore different tokenomics scenarios")
    
    # Set static simulation parameters
    simulation_months = 120  # 10 years
    total_supply = 2_000_000_000  # 2 billion tokens
    
    # TRAJECTORIES & SCENARIOS
    with st.sidebar.expander("Trajectories & Scenarios", expanded=True):
        st.markdown("**Configure token price and staking percentage trajectories**")

        # Price trajectory configuration
        price_model = st.selectbox(
            "Price Model",
            ["Linear Growth", "Constant", "Exponential Growth"],
            help="Choose how token price evolves over time. Constant keeps price fixed, Linear adds fixed amount monthly, Exponential compounds monthly."
        )

        initial_price = st.number_input(
            "Initial Token Price ($)",
            min_value=0.01, max_value=100.0, value=0.05, step=0.01,
            help="Starting price per token in USD at month 0."
        )

        if price_model == "Linear Growth":
            monthly_growth = st.number_input(
                "Monthly Price Increase ($)",
                min_value=0.0, max_value=1.0, value=0.001, step=0.001,
                format="%.3f",
                help="Fixed dollar amount added to price each month (e.g., 0.01 = +$0.01/month)."
            )
            price_trajectory = LinearPrice(initial_price, monthly_growth)

        elif price_model == "Exponential Growth":
            monthly_growth_rate = st.number_input(
                "Monthly Growth Rate (%)",
                min_value=0.0, max_value=20.0, value=2.0, step=0.1,
                help="Compound monthly growth rate as percentage (e.g., 2.0 = 2% monthly growth)."
            ) / 100
            price_trajectory = ExponentialPrice(initial_price, monthly_growth_rate)

        else:  # Constant
            price_trajectory = ConstantPrice(initial_price)

        # Show price trajectory preview
        if price_model != "Constant":
            preview_months = min(12, simulation_months)
            preview_prices = [price_trajectory.get_price(m) for m in range(preview_months)]
            st.caption(f"Price preview (12 months): ${preview_prices[0]:.2f} to ${preview_prices[-1]:.2f}")

        # Staking percentage trajectory configuration
        staking_percent = st.slider(
            "Staking Percentage (%)",
            min_value=0, max_value=100, value=30, step=1,
            help="Percentage of circulating supply staked each month."
        ) / 100
    
    # MODULE 1: REVENUE BUYBACKS
    with st.sidebar.expander("Module 1: Revenue Buybacks", expanded=False):
        st.markdown("**Protocol revenue to token buybacks to worker rewards**")
        
        mrr_usd = st.number_input(
            "Monthly Recurring Revenue ($)",
            min_value=10_000, max_value=10_000_000, value=100_000, step=10_000,
            help="Protocol's monthly revenue in USD. This revenue is used to buy back tokens from the market."
        )
        
        buyback_percentage = st.slider(
            "Revenue % Used for Buybacks",
            min_value=0.1, max_value=1.0, value=0.8, step=0.05,
            help="Percentage of monthly revenue allocated to token buybacks (default: 80%). Remaining goes to other uses."
        )
        
        worker_retention = st.slider(
            "Worker Token Retention Rate",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Percentage of bought tokens that workers keep vs. sell immediately. 0% = all tokens sold back, 100% = all tokens held."
        )
        
        # Show buyback calculation
        monthly_buyback_usd = mrr_usd * buyback_percentage
        st.caption(f"Monthly buyback budget: ${monthly_buyback_usd:,.0f}")
    
    # MODULE 2: STAKING REWARDS
    with st.sidebar.expander("Module 2: DPoS Staking Rewards", expanded=False):
        st.markdown("**Exponentially decaying staking rewards for validators**")
        
        staking_budget = st.slider(
            "Staking Budget (% of total supply)",
            min_value=0.05, max_value=0.40, value=0.15, step=0.01,
            help="Total percentage of token supply allocated to staking rewards over the entire emission schedule."
        )
        
        staking_half_life = st.slider(
            "Staking Emission Half-life (months)",
            min_value=6, max_value=72, value=36, step=3,
            help="Time for staking emissions to decrease by 50%. Longer half-life = more sustained rewards."
        )
        
        # Show staking calculations
        staking_total_budget = total_supply * staking_budget
        initial_monthly_emission = staking_total_budget * np.log(2) / staking_half_life
        st.caption(f"Total staking budget: {staking_total_budget/1e6:.0f}M tokens")
        st.caption(f"Initial monthly emission: {initial_monthly_emission/1e6:.1f}M tokens")
    
    # MODULE 3: PARTICIPANT REWARDS
    with st.sidebar.expander("Module 3: Participant Rewards", expanded=False):
        st.markdown("**FDV-milestone unlocked participant rewards**")
        
        participant_budget = st.slider(
            "Participant Budget (% of total supply)",
            min_value=0.05, max_value=0.30, value=0.15, step=0.01,
            help="Total percentage of token supply allocated to participant rewards."
        )
        
        participant_half_life = st.slider(
            "Participant Emission Half-life (months)",
            min_value=12, max_value=84, value=48, step=6,
            help="Half-life for exponential decay of participant reward emissions."
        )
        
        st.markdown("**FDV Unlock Milestones**")
        col1, col2 = st.columns(2)
        
        with col1:
            milestone_1 = st.number_input(
                "Milestone 1 FDV ($M)",
                min_value=10, max_value=1000, value=100, step=10,
                help="First FDV milestone for unlocking participant rewards."
            ) * 1_000_000
            
            milestone_2 = st.number_input(
                "Milestone 2 FDV ($M)",
                min_value=50, max_value=2000, value=300, step=50,
                help="Second FDV milestone for additional unlocks."
            ) * 1_000_000
            
            milestone_3 = st.number_input(
                "Milestone 3 FDV ($M)",
                min_value=100, max_value=5000, value=1000, step=100,
                help="Third FDV milestone for final unlocks."
            ) * 1_000_000
        
        with col2:
            unlock_1 = st.slider(
                "Unlock % at Milestone 1",
                min_value=0.1, max_value=1.0, value=0.25, step=0.05,
                help="Percentage of total budget unlocked at first milestone."
            )
            
            unlock_2 = st.slider(
                "Unlock % at Milestone 2",
                min_value=0.1, max_value=1.0, value=0.50, step=0.05,
                help="Cumulative percentage unlocked at second milestone."
            )
            
            unlock_3 = st.slider(
                "Unlock % at Milestone 3",
                min_value=0.5, max_value=1.0, value=1.0, step=0.05,
                help="Final percentage unlocked (typically 100%)."
            )
        
        fdv_milestones = [milestone_1, milestone_2, milestone_3]
        unlock_percentages = [unlock_1, unlock_2, unlock_3]
        
        # Show participant reward calculations
        participant_total_budget = total_supply * participant_budget
        st.caption(f"Total participant budget: {participant_total_budget/1e6:.0f}M tokens")
    
    # MODULE 4: MINING RESERVE
    with st.sidebar.expander("Mining Reserve", expanded=False):
        st.markdown("**Permanently locked reserve for future mining incentives**")
        
        mining_reserve_budget = st.slider(
            "Mining Reserve (% of total supply)",
            min_value=0.0, max_value=0.20, value=0.05, step=0.01,
            help="Percentage of token supply permanently locked for future mining incentives. This allocation is made immediately and never circulates."
        )
        
        # Show mining reserve calculations
        mining_reserve_total_budget = total_supply * mining_reserve_budget
        st.caption(f"Total mining reserve: {mining_reserve_total_budget/1e6:.0f}M tokens (permanently locked)")
    
    # MODULE 5: TESTNET ALLOCATION
    with st.sidebar.expander("Testnet Allocation", expanded=False):
        st.markdown("**Exponentially decaying testnet incentives**")
        
        testnet_allocation_budget = st.slider(
            "Testnet Allocation (% of total supply)",
            min_value=0.0, max_value=0.20, value=0.05, step=0.01,
            help="Total percentage of token supply allocated to testnet incentives over the entire emission schedule."
        )
        
        testnet_allocation_half_life = st.slider(
            "Testnet Emission Half-life (months)",
            min_value=12, max_value=84, value=72, step=6,
            help="Half-life for exponential decay of testnet allocation emissions."
        )
        
        # Show testnet allocation calculations
        testnet_allocation_total_budget = total_supply * testnet_allocation_budget
        initial_monthly_emission = testnet_allocation_total_budget * np.log(2) / testnet_allocation_half_life
        st.caption(f"Total testnet budget: {testnet_allocation_total_budget/1e6:.0f}M tokens")
        st.caption(f"Initial monthly emission: {initial_monthly_emission/1e6:.1f}M tokens")
    
    # DYNAMIC ALLOCATION VALIDATION
    total_dynamic_allocation = staking_budget + participant_budget + mining_reserve_budget + testnet_allocation_budget
    
    if total_dynamic_allocation > 0.40:
        st.sidebar.error(f"⚠️ Total dynamic allocation ({total_dynamic_allocation*100:.1f}%) exceeds 40% limit!")
        st.sidebar.markdown(f"""
        **Current allocations:**
        - Staking: {staking_budget*100:.1f}%
        - Participants: {participant_budget*100:.1f}%
        - Mining Reserve: {mining_reserve_budget*100:.1f}%
        - Testnet: {testnet_allocation_budget*100:.1f}%
        """)
        st.sidebar.markdown(f"**Remaining for static allocation: {(1.0 - total_dynamic_allocation)*100:.1f}%**")
    else:
        st.sidebar.success(f"✅ Total dynamic allocation: {total_dynamic_allocation*100:.1f}%")
        st.sidebar.markdown(f"**Remaining for static allocation: {(1.0 - total_dynamic_allocation)*100:.1f}%**")
    
    # Create configuration object
    config = SimulationConfig(
        total_supply=int(total_supply),
        simulation_months=simulation_months,
        mrr_usd=mrr_usd,
        buyback_percentage=buyback_percentage,
        worker_retention_rate=worker_retention,
        staking_budget_percentage=staking_budget,
        staking_half_life_months=staking_half_life,
        participant_budget_percentage=participant_budget,
        participant_half_life_months=participant_half_life,
        mining_reserve_percentage=mining_reserve_budget,
        mining_reserve_half_life_months=60,  # Not used since it's immediate allocation
        testnet_allocation_percentage=testnet_allocation_budget,
        testnet_allocation_half_life_months=testnet_allocation_half_life,
        fdv_milestones=fdv_milestones,
        unlock_percentages=unlock_percentages,
        staking_percent=staking_percent
    )
    
    return config, price_trajectory

def create_charts(results: dict) -> None:
    """
    Create comprehensive visualization charts from simulation results using Altair
    
    Args:
        results: Dictionary containing simulation results
    """
    
    # Configure Altair
    alt.data_transformers.enable('json')
    
    # Main metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    final_month = len(results['circulating_supply']) - 1
    
    with col1:
        st.metric(
            "Final Circulating Supply",
            f"{results['circulating_supply'][final_month]/1e6:.1f}M"
        )
    
    with col2:
        st.metric(
            "Average Monthly Net Flow",
            f"{np.mean(results['net_flow'])/1e6:.1f}M tokens"
        )
    
    with col3:
        st.metric(
            "Final Token Price",
            f"${results['prices'][final_month]:.3f}"
        )
    
    with col4:
        st.metric(
            "Final FDV",
            f"${results['config'].total_supply * results['prices'][final_month] / 1e6:.0f}M"
        )
    
    # Token Supply Breakdown Over Time plot
    st.subheader("Token Supply Breakdown Over Time")
    st.caption("""
    This stacked area chart shows the breakdown of the total TNT supply over time:
    - Circulating: TNT that is liquid and not staked
    - DPoS Staking: TNT that is staked and participating in network security
    - Early Bird (M3): TNT that is still locked and not yet released to the market
    The sum of all areas never exceeds the total supply cap.
    """)
    # Calculate mutually exclusive supply categories
    unlocked_unstaked = results['circulating_supply'] - results['staked_tokens']
    staked = results['staked_tokens']
    locked = results['module3_locked_tokens']
    burned = np.cumsum(results['module1_burned'])  # Cumulative burned tokens
    stacked_df = pd.DataFrame({
        'Month': results['months'],
        'Burned': burned / 1e6,
        'Circulating': unlocked_unstaked / 1e6,
        'DPoS Staking': staked / 1e6,
        'Early Bird (M3)': locked / 1e6
    })
    stacked_melted = stacked_df.melt(
        id_vars=['Month'],
        var_name='Type',
        value_name='TNT (Millions)'
    )
    stack_color_scale = alt.Scale(
        domain=['Burned', 'Circulating', 'DPoS Staking', 'Early Bird (M3)'],
        range=['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    )
    stacked_chart = alt.Chart(stacked_melted).mark_area(
        opacity=0.7,
        line={'strokeWidth': 2}
    ).encode(
        x=alt.X('Month:Q', title='Months'),
        y=alt.Y('TNT (Millions):Q', stack='zero', title='TNT (Millions)'),
        color=alt.Color('Type:N', scale=stack_color_scale),
        tooltip=[
            alt.Tooltip('Month:Q', title='Month'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('TNT (Millions):Q', title='TNT (M)', format='.2f')
        ]
    ).properties(
        height=400
    ).interactive()
    st.altair_chart(stacked_chart, use_container_width=True)

    # Monthly Net Flow plot
    st.subheader("Monthly Net Flow")
    st.caption("""
    This plot shows the net monthly change in circulating TNT supply. Positive values indicate net emissions (more TNT entering circulation), while negative values indicate net sinks (TNT being removed from circulation, e.g., via buybacks or retention).
    """)
    flow_df = pd.DataFrame({
        'Month': results['months'],
        'Net Flow (Millions)': results['net_flow'] / 1e6,
        'Flow Type': ['Positive' if x >= 0 else 'Negative' for x in results['net_flow']]
    })
    base = alt.Chart(flow_df)
    flow_chart = base.mark_area(
        opacity=0.7
    ).encode(
        x=alt.X('Month:Q', title='Months'),
        y=alt.Y('Net Flow (Millions):Q', title='Net Flow (Millions)'),
        color=alt.Color(
            'Flow Type:N',
            scale=alt.Scale(range=['#ff7f0e', '#d62728']),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('Month:Q', title='Month'),
            alt.Tooltip('Net Flow (Millions):Q', title='Net Flow (M)', format='.1f')
        ]
    )
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='gray',
        strokeDash=[5, 5],
        opacity=0.5
    ).encode(y='y:Q')
    combined_flow = (flow_chart + zero_line).properties(height=400).interactive()
    st.altair_chart(combined_flow, use_container_width=True)
    
    # Flow component breakdown
    st.subheader("Token Flow Component Analysis")
    
    # Create comprehensive DataFrame for all components
    module1_net = (results['module1_redistributions'] - 
                   results['module1_buybacks'] * results['config'].worker_retention_rate)
    
    # Cumulative Early Participation Emissions (unlocked and locked)
    cumulative_unlocked = np.cumsum(results['module3_unlocked']) / 1e6
    cumulative_locked = results['module3_locked_tokens'] / 1e6
    months = results['months']
    
    # Only update the Early Participation Emissions plot to be cumulative and show both unlocked and locked
    early_participation_df = pd.DataFrame({
        'Month': months,
        'Released': cumulative_unlocked,
        'Accumulated': cumulative_locked
    })
    early_participation_melted = early_participation_df.melt(
        id_vars=['Month'],
        var_name='Type',
        value_name='TNT (Millions)'
    )
    early_color_scale = alt.Scale(
        domain=['Released', 'Accumulated'],
        range=['#9467bd', '#bbbbbb']
    )
    early_participation_chart = alt.Chart(early_participation_melted).mark_line(strokeWidth=3).encode(
        x=alt.X('Month:Q', title='Months'),
        y=alt.Y('TNT (Millions):Q', title='TNT (Millions)'),
        color=alt.Color('Type:N', scale=early_color_scale),
        tooltip=[
            alt.Tooltip('Month:Q', title='Month'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('TNT (Millions):Q', title='TNT (M)', format='.2f')
        ]
    ).properties(
        width=300,
        height=350,
        title='Early Participation Emissions (Cumulative)'
    ).interactive()
    
    # Update the components_df for the other plots (leave Early Participation Emissions out)
    components_df = pd.DataFrame({
        'Month': results['months'],
        'Static Vesting': results['static_vesting_flow'] / 1e6,
        'Staking Emissions': results['module2_emissions'] / 1e6
    })
    components_melted = components_df.melt(
        id_vars=['Month'],
        var_name='Component',
        value_name='TNT (Millions)'
    )
    components_chart = alt.Chart(components_melted).mark_line(
        strokeWidth=2
    ).encode(
        x=alt.X('Month:Q', title='Months'),
        y=alt.Y('TNT (Millions):Q', title='TNT (Millions)'),
        color=alt.Color(
            'Component:N',
            scale=alt.Scale(range=['#1f77b4', '#2ca02c']),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('Month:Q', title='Month'),
            alt.Tooltip('Component:N', title='Component'),
            alt.Tooltip('TNT (Millions):Q', title='TNT (M)', format='.2f')
        ]
    ).properties(
        width=300,
        height=200
    ).facet(
        facet='Component:N',
        columns=2  # Facet by component, max 2 per row
    ).resolve_scale(
        y='independent'
    ).interactive()
    
    # Display the Early Participation Emissions cumulative plot and the other components chart
    st.caption("""
    This plot shows the cumulative sum of Early Participation (Module 3) emissions:
    - Released: TNT that has been unlocked and distributed to participants
    - Accumulated: TNT that has been emitted but is still locked, awaiting milestone release
    This helps visualize the total potential and actual distribution to early participants.
    """)
    st.altair_chart(early_participation_chart, use_container_width=True)
    st.caption("""
    The following plots break down the main token flow components:
    - Static Vesting: Scheduled token releases to core stakeholders
    - Staking Emissions: Monthly TNT distributed to DPoS stakers
    Each plot shows the monthly flow for its component.
    """)
    st.altair_chart(components_chart, use_container_width=True)
    
    # Protocol inflow plot (now cumulative)
    st.subheader("Cumulative Protocol Inflow (MRR to Protocol Treasury)")
    protocol_df = pd.DataFrame({
        'Month': results['months'],
        'Protocol Inflow (TNT)': np.cumsum(results['module1_protocol_inflow']) / 1e6
    })
    protocol_chart = alt.Chart(protocol_df).mark_area(
        color='#17becf',
        opacity=0.6
    ).encode(
        x=alt.X('Month:Q', title='Months'),
        y=alt.Y('Protocol Inflow (TNT):Q', title='Cumulative Protocol Inflow (Millions)'),
        tooltip=[
            alt.Tooltip('Month:Q', title='Month'),
            alt.Tooltip('Protocol Inflow (TNT):Q', title='Cumulative Protocol Inflow (M)', format='.2f')
        ]
    ).properties(
        height=350
    ).interactive()
    st.altair_chart(protocol_chart, use_container_width=True)

    # Price and FDV analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Token Price Trajectory")
        
        price_df = pd.DataFrame({
            'Month': results['months'],
            'Price ($)': results['prices']
        })
        
        price_chart = alt.Chart(price_df).mark_line(
            color='#2ca02c',
            strokeWidth=3
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Price ($):Q', title='Price ($)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Price ($):Q', title='Price', format='$.3f')
            ]
        ).properties(
            height=400
        ).interactive()
        
        st.altair_chart(price_chart, use_container_width=True)
    
    with col2:
        st.subheader("Fully Diluted Valuation")
        
        fdv_df = pd.DataFrame({
            'Month': results['months'],
            'FDV ($ Millions)': results['fdv'] / 1e6
        })
        
        # Base FDV chart
        fdv_chart = alt.Chart(fdv_df).mark_line(
            color='#d62728',
            strokeWidth=3
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('FDV ($ Millions):Q', title='FDV ($ Millions)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('FDV ($ Millions):Q', title='FDV ($M)', format='.0f')
            ]
        )
        
        # Create milestone lines
        config = results['config']
        milestone_data = []
        for i, milestone in enumerate(config.fdv_milestones):
            milestone_data.append({
                'Milestone': f"Milestone {i+1}: ${milestone/1e6:.0f}M ({config.unlock_percentages[i]*100:.0f}% unlock)",
                'Value': milestone / 1e6
            })
        
        milestone_df = pd.DataFrame(milestone_data)
        
        milestone_rules = alt.Chart(milestone_df).mark_rule(
            color='gray',
            strokeDash=[5, 5],
            opacity=0.7
        ).encode(
            y=alt.Y('Value:Q'),
            tooltip=['Milestone:N']
        )
        
        # Combine FDV chart with milestone lines
        combined_fdv = (fdv_chart + milestone_rules).properties(height=400).interactive()
        
        st.altair_chart(combined_fdv, use_container_width=True)
    
    # Data export section
    with st.expander("Export Simulation Data"):
        st.markdown("Download simulation results for further analysis")
        
        df = pd.DataFrame({
            'Month': results['months'],
            'Circulating_Supply': results['circulating_supply'],
            'Total_Supply': results['total_supply'],
            'Token_Price': results['prices'],
            'FDV': results['fdv'],
            'Net_Flow': results['net_flow'],
            'Static_Vesting_Flow': results['static_vesting_flow'],
            'Module1_Buybacks': results['module1_buybacks'],
            'Module1_Redistributions': results['module1_redistributions'],
            'Module1_Protocol_Inflow': results['module1_protocol_inflow'],
            'Module1_Burned': results['module1_burned'],
            'Module2_Emissions': results['module2_emissions'],
            'Module3_Unlocked': results['module3_unlocked']
        })
        
        st.dataframe(df.head(10), use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Complete Dataset (CSV)",
            data=csv,
            file_name="tokenomics_simulation_results.csv",
            mime="text/csv"
        )

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("Titan MacroEconomics Simulation")
    st.markdown("""
    **Analyze circulating supply and net flow dynamics under different tokenomics scenarios**
    
    This simulation models three dynamic emission modules plus static vesting schedules to understand 
    how token supply evolves over time. Adjust parameters in the sidebar to explore different scenarios.
    """)
    
    # Configuration sidebar
    config, price_trajectory = create_sidebar_config()
    
    # Add run button in sidebar
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_simulation:
        with st.spinner("🔄 Running tokenomics simulation..."):
            # Initialize and run simulation
            sim = TokenomicsSimulation(config, price_trajectory)
            results = sim.run_simulation()
            
            # Cache results in session state
            st.session_state.results = results
            st.session_state.has_results = True
    
    # Display results if available
    if hasattr(st.session_state, 'has_results') and st.session_state.has_results:
        st.success("✅ Simulation completed successfully!")
        
        # Show configuration summary
        with st.expander("Current Configuration Summary", expanded=False):
            results = st.session_state.results
            config = results['config']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**General**")
                st.write(f"Duration: {config.simulation_months} months")
                st.write(f"Total Supply: {config.total_supply/1e9:.1f}B tokens")
                st.write(f"Price Model: {results['price_model']}")
            
            with col2:
                st.markdown("**Module Budgets**")
                st.write(f"Staking: {config.staking_budget_percentage*100:.1f}% ({config.total_supply*config.staking_budget_percentage/1e6:.0f}M tokens)")
                st.write(f"Participants: {config.participant_budget_percentage*100:.1f}% ({config.total_supply*config.participant_budget_percentage/1e6:.0f}M tokens)")
                st.write(f"Mining Reserve: {config.mining_reserve_percentage*100:.1f}% ({config.total_supply*config.mining_reserve_percentage/1e6:.0f}M tokens)")
                st.write(f"Testnet: {config.testnet_allocation_percentage*100:.1f}% ({config.total_supply*config.testnet_allocation_percentage/1e6:.0f}M tokens)")
                st.write(f"Monthly Revenue: ${config.mrr_usd:,.0f}")
            
            with col3:
                st.markdown("**Half-lives**")
                st.write(f"Staking: {config.staking_half_life_months:.0f} months")
                st.write(f"Participants: {config.participant_half_life_months:.0f} months")
                st.write(f"Testnet: {config.testnet_allocation_half_life_months:.0f} months")
                st.write(f"Worker Retention: {config.worker_retention_rate*100:.1f}%")
        
        # Display charts
        create_charts(st.session_state.results)
        
    else:
        # Show getting started message
        st.info("Configure parameters in the sidebar and click Run Simulation to begin analysis")
        
        # Show sample configuration tips
        st.markdown("### Quick Start Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **For High Growth Scenarios:**
            - Set exponential price growth (3-5% monthly)
            - Lower worker retention (0-20%)
            - Shorter half-lives for faster distribution
            """)
        
        with col2:
            st.markdown("""
            **For Conservative Analysis:**
            - Use constant or linear price growth
            - Higher worker retention (50-80%)
            - Longer half-lives for sustained emissions
            """)

if __name__ == "__main__":
    main()
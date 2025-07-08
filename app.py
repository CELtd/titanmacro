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

def create_sidebar_config() -> tuple[SimulationConfig, object, float]:
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
        
        # Show buyback calculation
        monthly_buyback_usd = mrr_usd * buyback_percentage
        st.caption(f"Monthly buyback budget: ${monthly_buyback_usd:,.0f}")
        
        # Add constraint warning note
        st.info("""
        **⚠️ Important Constraint:** Buybacks and burns are limited by available circulating supply. 
        If revenue is too high relative to circulating tokens, the simulation will flag constraint violations.
        
        **Buy Pressure Metrics:** The simulation calculates buy pressure as percentage of circulating supply 
        and buyback efficiency (tokens per USD). Higher percentages indicate stronger buy pressure.
        """)
    
    # DYNAMIC ALLOCATION PERCENTAGES
    with st.sidebar.expander("Dynamic Allocation Percentages", expanded=True):
        st.markdown("**Configure the four dynamic allocation modules (max 40% total)**")
        
        # Staking Budget
        staking_budget = st.slider(
            "Staking Budget (% of total supply)",
            min_value=0.05, max_value=0.40, value=0.15, step=0.01,
            help="Total percentage of token supply allocated to staking rewards over the entire emission schedule."
        )
        
        # Participant Budget
        participant_budget = st.slider(
            "Participant Budget (% of total supply)",
            min_value=0.05, max_value=0.30, value=0.15, step=0.01,
            help="Total percentage of token supply allocated to participant rewards."
        )
        
        # Mining Reserve Budget
        mining_reserve_budget = st.slider(
            "Mining Reserve (% of total supply)",
            min_value=0.0, max_value=0.20, value=0.05, step=0.01,
            help="Percentage of token supply permanently locked for future mining incentives. This allocation is made immediately and never circulates."
        )
        
        # Testnet Allocation Budget
        testnet_allocation_budget = st.slider(
            "Testnet Allocation (% of total supply)",
            min_value=0.0, max_value=0.20, value=0.05, step=0.01,
            help="Total percentage of token supply allocated to testnet incentives over the entire emission schedule."
        )
        
        # Dynamic allocation validation
        total_dynamic_allocation = staking_budget + participant_budget + mining_reserve_budget + testnet_allocation_budget
        
        if total_dynamic_allocation > 0.40:
            st.error(f"⚠️ Total dynamic allocation ({total_dynamic_allocation*100:.1f}%) exceeds 40% limit!")
            st.markdown(f"""
            **Current allocations:**
            - Staking: {staking_budget*100:.1f}%
            - Participants: {participant_budget*100:.1f}%
            - Mining Reserve: {mining_reserve_budget*100:.1f}%
            - Testnet: {testnet_allocation_budget*100:.1f}%
            """)
            st.markdown(f"**Remaining for static allocation: {(1.0 - total_dynamic_allocation)*100:.1f}%**")
        else:
            st.success(f"✅ Total dynamic allocation: {total_dynamic_allocation*100:.1f}%")
            st.markdown(f"**Remaining for static allocation: {(1.0 - total_dynamic_allocation)*100:.1f}%**")
    
    # MODULE 2: STAKING REWARDS CONFIGURATION
    with st.sidebar.expander("Staking Rewards Configuration", expanded=False):
        st.markdown("**Configure staking emission parameters**")
        
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
    
    # MODULE 3: PARTICIPANT REWARDS CONFIGURATION
    with st.sidebar.expander("Participant Rewards Configuration", expanded=False):
        st.markdown("**Configure participant reward parameters and FDV milestones**")
        
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
    
    # MODULE 5: TESTNET ALLOCATION CONFIGURATION
    with st.sidebar.expander("Testnet Allocation Configuration", expanded=False):
        st.markdown("**Configure testnet emission parameters**")
        
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
    
    # FIRST YEAR ANALYSIS CONFIGURATION
    with st.sidebar.expander("First Year Analysis Configuration", expanded=False):
        st.markdown("**Configure assumptions for first year cumulative analysis**")
        
        excess_buyback_percentage = st.slider(
            "Excess Token Buyback Percentage (%)",
            min_value=0, max_value=50, value=20, step=5,
            help="Percentage of excess tokens (positive net flow) that are bought back to maintain price stability. 100% means all excess tokens are bought back, 20% means only 20% of excess tokens are bought back."
        ) / 100
    
    # Create configuration object
    config = SimulationConfig(
        total_supply=int(total_supply),
        simulation_months=simulation_months,
        mrr_usd=mrr_usd,
        buyback_percentage=buyback_percentage,
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
    
    return config, price_trajectory, excess_buyback_percentage

def create_charts(results: dict, excess_buyback_percentage: float = 1.0) -> None:
    """
    Create comprehensive visualization charts from simulation results using Altair
    
    Args:
        results: Dictionary containing simulation results
        excess_buyback_percentage: Percentage of excess tokens to buy back (0.0 to 1.0)
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
    
    # # NEW: Buy pressure metrics row
    # if hasattr(st.session_state, 'constraint_analysis'):
    #     buy_pressure = st.session_state.constraint_analysis['buy_pressure_metrics']
        
    #     col1, col2, col3, col4 = st.columns(4)
        
    #     with col1:
    #         st.metric(
    #             "Avg Buyback Impact",
    #             f"{buy_pressure['avg_buyback_pct_supply']:.2f}% of supply"
    #         )
        
    #     with col2:
    #         st.metric(
    #             "Peak Buyback Impact", 
    #             f"{buy_pressure['max_buyback_pct_supply']:.2f}% of supply"
    #         )
        
    #     with col3:
    #         st.metric(
    #             "Total Buyback Volume",
    #             f"${buy_pressure['total_buyback_usd']/1e6:.1f}M USD"
    #         )
        
    #     with col4:
    #         st.metric(
    #             "Buyback Efficiency",
    #             f"{buy_pressure['avg_buyback_efficiency']:.1f} tokens/USD"
    #         )
    
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

    # NEW: Circulating Supply Components Breakdown
    st.subheader("Circulating Supply Components Breakdown")
    st.caption("""
    This stacked area chart shows the detailed breakdown of what contributes to and reduces circulating supply over time.
    Positive components (additive) are shown above the zero line, negative components (subtractive) are shown below.
    This helps identify the main drivers of inflation or deflation in the circulating supply.
    """)
    
    # Calculate all the components that affect circulating supply
    # Additive components (positive contributions to circulating supply)
    static_vesting_cumulative = np.cumsum(results['static_vesting_flow']) / 1e6
    staking_emissions_cumulative = np.cumsum(results['module2_emissions']) / 1e6
    participant_unlocked_cumulative = np.cumsum(results['module3_unlocked']) / 1e6
    testnet_emissions_cumulative = np.cumsum(results['testnet_allocation_emissions']) / 1e6
    
    # Subtractive components (negative contributions to circulating supply)
    burned_tokens_cumulative = np.cumsum(results['module1_burned']) / 1e6
    
    # Create DataFrame for the components breakdown
    components_breakdown_df = pd.DataFrame({
        'Month': results['months'],
        'Static Vesting': static_vesting_cumulative,
        'Staking Emissions': staking_emissions_cumulative,
        'Participant Unlocks': participant_unlocked_cumulative,
        'Testnet Emissions': testnet_emissions_cumulative,
        'Burned Tokens': -burned_tokens_cumulative  # Negative to show below zero line
    })
    
    # Melt the DataFrame for Altair
    components_melted = components_breakdown_df.melt(
        id_vars=['Month'],
        var_name='Component',
        value_name='TNT (Millions)'
    )
    
    # Create color scale with positive components in blues/greens, negative in reds
    component_color_scale = alt.Scale(
        domain=['Static Vesting', 'Staking Emissions', 'Participant Unlocks', 'Testnet Emissions', 'Burned Tokens'],
        range=['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#d62728']
    )
    
    # Create the stacked area chart
    components_chart = alt.Chart(components_melted).mark_area(
        opacity=0.7,
        line={'strokeWidth': 1}
    ).encode(
        x=alt.X('Month:Q', title='Months'),
        y=alt.Y('TNT (Millions):Q', stack='zero', title='TNT (Millions)'),
        color=alt.Color('Component:N', scale=component_color_scale),
        tooltip=[
            alt.Tooltip('Month:Q', title='Month'),
            alt.Tooltip('Component:N', title='Component'),
            alt.Tooltip('TNT (Millions):Q', title='TNT (M)', format='.2f')
        ]
    ).properties(
        height=400
    ).interactive()
    
    st.altair_chart(components_chart, use_container_width=True)

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
    
    # First Year Cumulative Analysis (side by side plots)
    st.subheader("First Year Cumulative Analysis")
    buyback_percentage_display = excess_buyback_percentage * 100
    st.caption(f"""
    Analysis of the first 12 months showing the change in circulating supply from TGE baseline and the USD cost to buy back {buyback_percentage_display:.0f}% of excess tokens to maintain price stability.
    """)
    
    # Get first 12 months of data
    first_year_months = results['months'][:12]
    first_year_net_flow = results['net_flow'][:12]
    first_year_prices = results['prices'][:12]
    first_year_circulating_supply = results['circulating_supply'][:12]
    
    # Calculate cumulative net flow starting from initial circulating supply (TGE baseline)
    initial_circulating_supply = first_year_circulating_supply[0] / 1e6  # Convert to millions
    cumulative_net_flow = (first_year_circulating_supply - initial_circulating_supply) / 1e6  # Delta from TGE baseline
    
    # Calculate monthly USD buyback cost (for right plot)
    # Only buy back positive net flow (excess supply) based on configured percentage
    monthly_buyback_tokens = np.where(first_year_net_flow > 0, first_year_net_flow * excess_buyback_percentage, 0) / 1e6  # Convert to millions
    monthly_usd_buyback_cost = monthly_buyback_tokens * 1e6 * first_year_prices / 1e6  # Convert to millions USD
    
    # Calculate cumulative USD buyback cost (alternative approach)
    cumulative_usd_buyback_cost = np.cumsum(monthly_usd_buyback_cost)
    
    # Create side-by-side plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Change in Circulating Supply from TGE (First 12 Months)**")
        cumulative_df = pd.DataFrame({
            'Month': first_year_months,
            'Change from TGE (Millions)': cumulative_net_flow
        })
        
        cumulative_chart = alt.Chart(cumulative_df).mark_line(
            color='#1f77b4',
            strokeWidth=3
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Change from TGE (Millions):Q', title='Change from TGE (Millions)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Change from TGE (Millions):Q', title='Change from TGE (M)', format='.2f')
            ]
        ).properties(
            height=300
        ).interactive()
        
        st.altair_chart(cumulative_chart, use_container_width=True)
    
    with col2:
        buyback_percentage_display = excess_buyback_percentage * 100
        st.markdown(f"**Monthly USD Cost to Buy Back {buyback_percentage_display:.0f}% of Excess Tokens**")
        usd_df = pd.DataFrame({
            'Month': first_year_months,
            'Monthly USD Cost (Millions)': monthly_usd_buyback_cost
        })
        
        usd_chart = alt.Chart(usd_df).mark_line(
            color='#d62728',
            strokeWidth=3
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Monthly USD Cost (Millions):Q', title='USD Cost (Millions)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Monthly USD Cost (Millions):Q', title='Monthly Cost ($M)', format='.2f')
            ]
        ).properties(
            height=300
        ).interactive()
        
        st.altair_chart(usd_chart, use_container_width=True)
    
    # Calculate and display cumulative buyback summary for first 6 months
    first_6_months_buyback_cost = np.sum(monthly_usd_buyback_cost[:6])
    first_6_months_buyback_tokens = np.sum(monthly_buyback_tokens[:6])
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"6-Month Cumulative Buyback Cost",
            f"${first_6_months_buyback_cost:.2f}M",
            help=f"Total USD cost to buy back {buyback_percentage_display:.0f}% of excess tokens over the first 6 months"
        )
    
    with col2:
        st.metric(
            f"6-Month Cumulative Buyback Volume",
            f"{first_6_months_buyback_tokens:.2f}M tokens",
            help=f"Total tokens bought back over the first 6 months at {buyback_percentage_display:.0f}% of excess supply"
        )
    
    with col3:
        avg_monthly_cost = first_6_months_buyback_cost / 6
        st.metric(
            f"Average Monthly Buyback Cost",
            f"${avg_monthly_cost:.2f}M",
            help=f"Average monthly USD cost over the first 6 months"
        )
    
    # Flow component breakdown
    st.subheader("Token Flow Component Analysis")
    
    # Create comprehensive DataFrame for all components
    # Note: module1_net calculation removed since worker_retention_rate was eliminated
    
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
    
    # NEW: Buyback/Burn Constraint Analysis Chart
    if hasattr(st.session_state, 'constraint_analysis'):
        st.subheader("Buyback/Burn Constraint Analysis")
        st.caption("""
        This chart shows the maximum sustainable monthly revenue based on available circulating supply.
        If your current revenue exceeds this line, buybacks and burns will be constrained by token availability.
        """)
        
        constraint_analysis = st.session_state.constraint_analysis
        
        # Create constraint analysis DataFrame
        constraint_df = pd.DataFrame({
            'Month': results['months'],
            'Max Sustainable Revenue ($K)': constraint_analysis['max_sustainable_revenue'] / 1000,
            'Current Revenue ($K)': constraint_analysis['current_revenue'] / 1000
        })
        
        # Base chart for max sustainable revenue
        base = alt.Chart(constraint_df)
        
        # Max sustainable revenue line
        max_revenue_chart = base.mark_line(
            color='#2ca02c',
            strokeWidth=3
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Max Sustainable Revenue ($K):Q', title='Revenue ($K)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Max Sustainable Revenue ($K):Q', title='Max Revenue ($K)', format='.0f')
            ]
        )
        
        # Current revenue line
        current_revenue_chart = base.mark_line(
            color='#d62728',
            strokeWidth=2,
            strokeDash=[5, 5]
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Current Revenue ($K):Q', title='Revenue ($K)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Current Revenue ($K):Q', title='Current Revenue ($K)', format='.0f')
            ]
        )
        
        # Combine charts
        constraint_chart = (max_revenue_chart + current_revenue_chart).properties(
            height=350
        ).interactive()
        
        st.altair_chart(constraint_chart, use_container_width=True)
        
        # NEW: Buy Pressure Over Time Chart
        st.subheader("Buy Pressure Analysis Over Time")
        st.caption("""
        This chart shows how buyback pressure varies over time, measured as percentage of circulating supply.
        Higher percentages indicate stronger buy pressure relative to available tokens.
        """)
        
        buy_pressure = constraint_analysis['buy_pressure_metrics']
        
        buy_pressure_df = pd.DataFrame({
            'Month': results['months'],
            'Buyback % of Supply': buy_pressure['monthly_buyback_percentage_of_supply'],
            'Buyback Volume (USD)': buy_pressure['monthly_buyback_volume_usd'] / 1000  # Convert to thousands
        })
        
        # Create dual-axis chart for buyback percentage and volume
        base = alt.Chart(buy_pressure_df)
        
        # Buyback percentage line
        percentage_chart = base.mark_line(
            color='#2ca02c',
            strokeWidth=3
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Buyback % of Supply:Q', title='Buyback % of Supply', scale=alt.Scale(domain=[0, None])),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Buyback % of Supply:Q', title='Buyback % of Supply', format='.2f')
            ]
        )
        
        # Buyback volume bars
        volume_chart = base.mark_bar(
            color='#1f77b4',
            opacity=0.6
        ).encode(
            x=alt.X('Month:Q', title='Months'),
            y=alt.Y('Buyback Volume (USD):Q', title='Buyback Volume ($K)'),
            tooltip=[
                alt.Tooltip('Month:Q', title='Month'),
                alt.Tooltip('Buyback Volume (USD):Q', title='Buyback Volume ($K)', format='.0f')
            ]
        )
        
        # Combine charts
        buy_pressure_chart = (percentage_chart + volume_chart).properties(
            height=350
        ).interactive()
        
        st.altair_chart(buy_pressure_chart, use_container_width=True)

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
    config, price_trajectory, excess_buyback_percentage = create_sidebar_config()
    
    # Add run button in sidebar
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_simulation:
        with st.spinner("🔄 Running tokenomics simulation..."):
            # Initialize and run simulation
            sim = TokenomicsSimulation(config, price_trajectory)
            results = sim.run_simulation()
            
            # Analyze buyback constraints
            constraint_analysis = sim.analyze_buyback_constraints()
            
            # Cache results in session state
            st.session_state.results = results
            st.session_state.constraint_analysis = constraint_analysis
            st.session_state.has_results = True
    
    # Display results if available
    if hasattr(st.session_state, 'has_results') and st.session_state.has_results:
        st.success("✅ Simulation completed successfully!")
        
        # NEW: Display buyback/burn constraint warnings
        constraint_analysis = st.session_state.constraint_analysis
        
        if constraint_analysis['revenue_constrained']:
            current_revenue = "${:,.0f}".format(constraint_analysis['current_revenue'])
            max_revenue = "${:,.0f}".format(constraint_analysis['min_sustainable_revenue'])
            st.warning("⚠️ **Revenue Constraint Warning**")
            st.write(f"Your current monthly revenue ({current_revenue}) exceeds the maximum sustainable revenue ({max_revenue}) based on available circulating supply.")
            st.write("**Impact:**")
            st.write(f"- {constraint_analysis['total_buyback_violations']} months have buyback constraint violations")
            st.write(f"- {constraint_analysis['total_burn_violations']} months have burn constraint violations")
            st.write(f"- The bottleneck occurs at month {constraint_analysis['constraint_bottleneck_month']}")
            st.write("**Recommendation:** Consider reducing monthly revenue or adjusting tokenomics parameters to ensure buybacks and burns don't exceed available circulating supply.")
            
            # NEW: Buy pressure analysis
            buy_pressure = constraint_analysis['buy_pressure_metrics']
            st.write("**Buy Pressure Analysis:**")
            st.write(f"- Average buyback impact: {buy_pressure['avg_buyback_pct_supply']:.2f}% of circulating supply")
            st.write(f"- Peak buyback impact: {buy_pressure['max_buyback_pct_supply']:.2f}% of circulating supply")
            st.write(f"- Total buyback volume: ${buy_pressure['total_buyback_usd']/1e6:.1f}M USD over {len(st.session_state.results['months'])} months")
            st.write(f"- Buyback efficiency: {buy_pressure['avg_buyback_efficiency']:.2f} tokens per USD")
        elif constraint_analysis['total_buyback_violations'] > 0 or constraint_analysis['total_burn_violations'] > 0:
            max_revenue = "${:,.0f}".format(constraint_analysis['min_sustainable_revenue'])
            st.info("ℹ️ **Constraint Analysis**")
            st.write("Some months have constraint violations but revenue is within sustainable limits:")
            st.write(f"- {constraint_analysis['total_buyback_violations']} months with buyback violations")
            st.write(f"- {constraint_analysis['total_burn_violations']} months with burn violations")
            st.write(f"- Maximum sustainable revenue: {max_revenue}")
            
            # NEW: Buy pressure analysis
            buy_pressure = constraint_analysis['buy_pressure_metrics']
            st.write("**Buy Pressure Analysis:**")
            st.write(f"- Average buyback impact: {buy_pressure['avg_buyback_pct_supply']:.2f}% of circulating supply")
            st.write(f"- Peak buyback impact: {buy_pressure['max_buyback_pct_supply']:.2f}% of circulating supply")
            st.write(f"- Total buyback volume: ${buy_pressure['total_buyback_usd']/1e6:.1f}M USD over {len(st.session_state.results['months'])} months")
            st.write(f"- Buyback efficiency: {buy_pressure['avg_buyback_efficiency']:.2f} tokens per USD")
        else:
            current_revenue = "${:,.0f}".format(constraint_analysis['current_revenue'])
            max_revenue = "${:,.0f}".format(constraint_analysis['min_sustainable_revenue'])
            st.success("✅ **No Constraint Violations**")
            st.write(f"Your current revenue ({current_revenue}) is less than sustainable limits.")
            st.write(f"Maximum sustainable revenue: {max_revenue}")
            
            # NEW: Buy pressure analysis
            buy_pressure = constraint_analysis['buy_pressure_metrics']
            st.write("**Buy Pressure Analysis:**")
            st.write(f"- Average buyback impact: {buy_pressure['avg_buyback_pct_supply']:.2f}% of circulating supply")
            st.write(f"- Peak buyback impact: {buy_pressure['max_buyback_pct_supply']:.2f}% of circulating supply")
            st.write(f"- Total buyback volume: ${buy_pressure['total_buyback_usd']/1e6:.1f}M USD over {len(st.session_state.results['months'])} months")
            st.write(f"- Buyback efficiency: {buy_pressure['avg_buyback_efficiency']:.2f} tokens per USD")
        
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
        
        # Display charts
        create_charts(st.session_state.results, excess_buyback_percentage)
        
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
            - Shorter half-lives for faster distribution
            """)
        
        with col2:
            st.markdown("""
            **For Conservative Analysis:**
            - Use constant or linear price growth
            - Longer half-lives for sustained emissions
            """)

if __name__ == "__main__":
    main()
"""
Core Tokenomics Simulation Module

This module contains the core simulation logic for TNT tokenomics modeling.
It includes price trajectory models, simulation configuration, and the main
simulation engine that tracks circulating supply and token flows.
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class SimulationConfig:
    """Configuration parameters for tokenomics simulation"""
    
    # General parameters
    total_supply: int = 2_000_000_000  # 2B tokens
    simulation_months: int = 60  # 5 years default
    
    # Module 1 - Revenue Buybacks
    mrr_usd: float = 100_000  # Monthly recurring revenue in USD
    buyback_percentage: float = 0.80  # Percentage of revenue used for buybacks
    worker_retention_rate: float = 0.00  # Percentage of bought tokens retained (not redistributed)
    
    # Module 2 - DPoS Staking Rewards
    staking_budget_percentage: float = 0.20  # Percentage of total supply allocated to staking
    staking_half_life_months: float = 36  # Half-life for exponential decay in months
    
    # Staking percent (fraction of circulating supply staked)
    staking_percent: float = 0.3  # Default 30% staked
    
    # Module 3 - Participant Rewards
    participant_budget_percentage: float = 0.15  # Percentage of total supply for participant rewards
    participant_half_life_months: float = 48  # Half-life for exponential decay in months
    
    # FDV Milestones for Module 3 unlocks
    fdv_milestones: List[float] = None  # FDV thresholds for unlocking tokens
    unlock_percentages: List[float] = None  # Percentage of total budget unlocked at each milestone
    
    def __post_init__(self):
        """Set default milestone values if not provided"""
        if self.fdv_milestones is None:
            self.fdv_milestones = [100_000_000, 300_000_000, 1_000_000_000]  # $100M, $300M, $1B
        if self.unlock_percentages is None:
            self.unlock_percentages = [0.25, 0.50, 1.00]  # 25%, 50%, 100%


class PriceTrajectory(ABC):
    """Abstract base class for price trajectory models"""
    
    @abstractmethod
    def get_price(self, month: int) -> float:
        """Get token price for a given month"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of the price model"""
        pass


class ConstantPrice(PriceTrajectory):
    """Constant price trajectory - price remains the same throughout simulation"""
    
    def __init__(self, price: float):
        """
        Initialize with constant price
        
        Args:
            price: Fixed token price in USD
        """
        self.price = price
    
    def get_price(self, month: int) -> float:
        return self.price
    
    def get_description(self) -> str:
        return f"Constant price: ${self.price:.2f}"


class LinearPrice(PriceTrajectory):
    """Linear price growth trajectory - price increases by fixed amount each month"""
    
    def __init__(self, initial_price: float, monthly_growth: float):
        """
        Initialize with linear growth parameters
        
        Args:
            initial_price: Starting token price in USD
            monthly_growth: Fixed monthly price increase in USD
        """
        self.initial_price = initial_price
        self.monthly_growth = monthly_growth
    
    def get_price(self, month: int) -> float:
        return max(0, self.initial_price + (month * self.monthly_growth))
    
    def get_description(self) -> str:
        return f"Linear growth: ${self.initial_price:.2f} + ${self.monthly_growth:.3f}/month"


class ExponentialPrice(PriceTrajectory):
    """Exponential price growth trajectory - price grows at compound rate each month"""
    
    def __init__(self, initial_price: float, monthly_growth_rate: float):
        """
        Initialize with exponential growth parameters
        
        Args:
            initial_price: Starting token price in USD
            monthly_growth_rate: Monthly compound growth rate (e.g., 0.02 for 2%)
        """
        self.initial_price = initial_price
        self.monthly_growth_rate = monthly_growth_rate
    
    def get_price(self, month: int) -> float:
        return self.initial_price * (1 + self.monthly_growth_rate) ** month
    
    def get_description(self) -> str:
        return f"Exponential growth: ${self.initial_price:.2f} × (1 + {self.monthly_growth_rate:.1%})^month"


class TokenomicsSimulation:
    """
    Core tokenomics simulation engine
    
    This class handles the monthly simulation of token supply dynamics,
    including static vesting schedules and three dynamic emission modules.
    """
    
    def __init__(self, config: SimulationConfig, price_trajectory: PriceTrajectory):
        """
        Initialize simulation with configuration and price model
        
        Args:
            config: Simulation configuration parameters
            price_trajectory: Price model to use for simulation
        """
        self.config = config
        self.price_trajectory = price_trajectory
        
        # Load static vesting data
        self.static_vesting_data = self._load_static_vesting_data()
        
        # Initialize results storage
        self.results = {}
        
    def _load_static_vesting_data(self) -> pd.DataFrame:
        """Load and process static vesting schedule data from quarterly milestones"""
        
        # Static vesting data provided (quarterly, cumulative)
        data = {
            'Quarter': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'Titan_Labs': [8333333,33333333,58333333,83333333,108333333,133333333,158333333,183333333,208333333,233333333,253333333,283333333,308333333,333333333,358333333,383333333,400000000,400000000,400000000,400000000,400000000],
            'Titan_Foundation': [4166667,16666667,29166667,41666667,54166667,66666667,79166667,91666667,104166667,116666667,129166667,141666667,154166667,166666667,179166667,191666667,200000000,200000000,200000000,200000000,200000000],
            'Seed_Fundraising': [944444,3777778,6611111,9444444,12277778,15111111,17944444,20777778,23611111,26444444,29277778,32111111,34000000,34000000,34000000,34000000,34000000,34000000,34000000,34000000,34000000],
            'SeriesA_Fundraising': [4611111,18444444,32277778,46111111,59944444,73777778,87611111,101444444,115277778,129111111,142944444,156777778,166000000,166000000,166000000,166000000,166000000,166000000,166000000,166000000,166000000],
            'Ecosystem': [2083333,8333333,14583333,20833333,27083333,33333333,39583333,45833333,52083333,58333333,64583333,70833333,77083333,83333333,89583333,95833333,100000000,100000000,100000000,100000000,100000000],
            'Testnet': [7500000,30000000,52500000,75000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000],
            'Market_Making': [7500000,30000000,52500000,75000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000,90000000],
            'Builder': [0,1500000,3000000,4500000,6000000,7500000,9000000,10500000,12000000,13500000,15000000,16500000,18000000,19500000,21000000,22500000,24000000,25500000,27000000,28500000,30000000],
            'RPGF': [0,1500000,3000000,4500000,6000000,7500000,9000000,10500000,12000000,13500000,15000000,16500000,18000000,19500000,21000000,22500000,24000000,25500000,27000000,28500000,30000000]
        }
        
        return pd.DataFrame(data)
    
    def _interpolate_quarterly_to_monthly(self, quarterly_data: pd.Series) -> np.ndarray:
        """
        Linearly interpolate quarterly cumulative data to monthly values
        
        Args:
            quarterly_data: Pandas series with quarterly cumulative values
            
        Returns:
            Array of monthly cumulative values
        """
        max_months = self.config.simulation_months
        months = np.arange(max_months)
        monthly_values = np.zeros(max_months)
        
        for i, month in enumerate(months):
            quarter = month // 3
            month_in_quarter = month % 3
            
            if quarter >= len(quarterly_data) - 1:
                # Use last quarter value for months beyond available data
                monthly_values[i] = quarterly_data.iloc[-1]
            else:
                # Linear interpolation between current and next quarter
                start_val = quarterly_data.iloc[quarter]
                end_val = quarterly_data.iloc[quarter + 1]
                progress = month_in_quarter / 3.0
                monthly_values[i] = start_val + (end_val - start_val) * progress
                
        return monthly_values
    
    def _calculate_exponential_decay_emission(self, month: int, total_budget: float, half_life_months: float) -> float:
        """
        Calculate exponential decay emission for a given month
        
        Uses formula: emission = budget × (ln(2) / half_life) × e^(-ln(2) × month / half_life)
        
        Args:
            month: Current month (0-indexed)
            total_budget: Total budget to be distributed
            half_life_months: Half-life parameter in months
            
        Returns:
            Token emission amount for this month
        """
        decay_rate = math.log(2) / half_life_months
        return total_budget * decay_rate * math.exp(-decay_rate * month)
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the complete tokenomics simulation
        
        Returns:
            Dictionary containing simulation results with monthly time series data
        """
        months = self.config.simulation_months
        
        # Initialize tracking arrays
        circulating_supply = np.zeros(months)
        total_supply_over_time = np.zeros(months)
        prices = np.zeros(months)
        fdv = np.zeros(months)
        
        # Module-specific tracking arrays
        static_vesting_flow = np.zeros(months)
        module1_buybacks = np.zeros(months)
        module1_redistributions = np.zeros(months)
        module1_protocol_inflow = np.zeros(months)  # NEW: protocol inflow
        module1_burned = np.zeros(months)           # NEW: burned tokens
        module2_emissions = np.zeros(months)
        module3_emissions = np.zeros(months)
        module3_unlocked = np.zeros(months)
        module3_locked_balance = 0
        
        # New: Staking percent vector and locked tokens tracking
        staking_percent_vec = np.full(months, self.config.staking_percent)
        staked_tokens = np.zeros(months)
        module3_locked_tokens = np.zeros(months)
        total_locked_tokens = np.zeros(months)
        
        # Track which FDV milestones have been reached
        module3_unlock_reached = [False] * len(self.config.fdv_milestones)
        
        # Interpolate static vesting schedules to monthly resolution
        static_monthly = {}
        for col in self.static_vesting_data.columns:
            if col != 'Quarter':
                static_monthly[col] = self._interpolate_quarterly_to_monthly(
                    self.static_vesting_data[col]
                )
        
        # Calculate total static vesting per month (cumulative)
        total_static_monthly = np.zeros(months)
        for month in range(months):
            if month < len(static_monthly['Titan_Labs']):
                month_total = sum(static_monthly[col][month] for col in static_monthly.keys())
                total_static_monthly[month] = month_total
            else:
                # Extend with last known values beyond available data
                total_static_monthly[month] = total_static_monthly[month-1] if month > 0 else 0
        
        # Convert cumulative to flow (monthly differences)
        for month in range(months):
            if month == 0:
                static_vesting_flow[month] = total_static_monthly[month]
            else:
                static_vesting_flow[month] = total_static_monthly[month] - total_static_monthly[month-1]
        
        # Calculate module budgets
        module2_total_budget = self.config.total_supply * self.config.staking_budget_percentage
        module3_total_budget = self.config.total_supply * self.config.participant_budget_percentage
        
        # Main simulation loop - process each month
        for month in range(months):
            # Get current token price
            prices[month] = self.price_trajectory.get_price(month)
            
            # Calculate total supply (static vesting + unlocked dynamic emissions)
            total_supply_over_time[month] = (
                total_static_monthly[month] +
                np.sum(module2_emissions[:month+1]) +
                np.sum(module3_unlocked[:month+1])
            )
            
            # Calculate Fully Diluted Valuation
            fdv[month] = total_supply_over_time[month] * prices[month]
            
            # MODULE 1: Revenue buybacks, protocol, and burns
            monthly_revenue = self.config.mrr_usd
            buyback_amount_usd = monthly_revenue * self.config.buyback_percentage
            protocol_amount_usd = monthly_revenue * 0.10  # 10% to protocol
            burn_amount_usd = monthly_revenue * 0.10      # 10% burned
            
            tokens_bought = buyback_amount_usd / prices[month] if prices[month] > 0 else 0
            tokens_protocol = protocol_amount_usd / prices[month] if prices[month] > 0 else 0
            tokens_burned = burn_amount_usd / prices[month] if prices[month] > 0 else 0
            
            module1_buybacks[month] = tokens_bought
            module1_protocol_inflow[month] = tokens_protocol
            module1_burned[month] = tokens_burned
            # Redistribute tokens not retained by workers
            module1_redistributions[month] = tokens_bought * (1 - self.config.worker_retention_rate)
            
            # MODULE 2: DPoS staking rewards (exponential decay)
            module2_emissions[month] = self._calculate_exponential_decay_emission(
                month, module2_total_budget, self.config.staking_half_life_months
            )
            
            # MODULE 3: Participant rewards with FDV-based unlocking
            # Calculate emission (goes to locked pool first)
            module3_emission = self._calculate_exponential_decay_emission(
                month, module3_total_budget, self.config.participant_half_life_months
            )
            module3_emissions[month] = module3_emission
            
            # Add emission to locked balance
            module3_locked_balance += module3_emission
            
            # Track module 3 locked tokens for this month
            module3_locked_tokens[month] = module3_locked_balance
            
            # Check FDV milestones and unlock tokens
            unlocked_this_month = 0
            for i, milestone in enumerate(self.config.fdv_milestones):
                if fdv[month] >= milestone and not module3_unlock_reached[i]:
                    # Milestone reached for first time
                    module3_unlock_reached[i] = True
                    unlock_percentage = self.config.unlock_percentages[i]
                    
                    # Calculate total amount that should be unlocked at this milestone
                    total_should_be_unlocked = module3_total_budget * unlock_percentage
                    already_unlocked = np.sum(module3_unlocked[:month])
                    
                    # Unlock additional tokens (limited by locked balance)
                    additional_unlock = max(0, total_should_be_unlocked - already_unlocked)
                    additional_unlock = min(additional_unlock, module3_locked_balance)
                    
                    unlocked_this_month += additional_unlock
                    module3_locked_balance -= additional_unlock
            
            module3_unlocked[month] = unlocked_this_month
            
            # Calculate circulating supply
            if month == 0:
                circulating_supply[month] = static_vesting_flow[month]
            else:
                circulating_supply[month] = circulating_supply[month-1] + static_vesting_flow[month]
            
            # Add dynamic emissions to circulation
            circulating_supply[month] += module2_emissions[month]  # Staking rewards
            circulating_supply[month] += module3_unlocked[month]  # Unlocked participant rewards
            circulating_supply[month] += module1_redistributions[month]  # Redistributed buybacks
            circulating_supply[month] += module1_protocol_inflow[month]  # Protocol inflow (remains circulating)
            
            # Subtract tokens retained by workers (temporarily out of circulation)
            circulating_supply[month] -= module1_buybacks[month] * self.config.worker_retention_rate
            # Subtract burned tokens (permanently removed)
            circulating_supply[month] -= module1_burned[month]
            
            # Calculate staked tokens for this month (after all emissions)
            staked_tokens[month] = circulating_supply[month] * staking_percent_vec[month]
            # Total locked = staked + module3 locked
            total_locked_tokens[month] = staked_tokens[month] + module3_locked_tokens[month]
        
        # Calculate net flow (positive = net emission, negative = net sink)
        net_flow = (
            static_vesting_flow +
            module2_emissions +
            module3_unlocked +
            module1_redistributions +
            module1_protocol_inflow -
            module1_burned -
            (module1_buybacks * self.config.worker_retention_rate)
        )
        
        # Store comprehensive results
        self.results = {
            # Time series data
            'months': np.arange(months),
            'circulating_supply': circulating_supply,
            'total_supply': total_supply_over_time,
            'prices': prices,
            'fdv': fdv,
            'net_flow': net_flow,
            
            # Flow components
            'static_vesting_flow': static_vesting_flow,
            'module1_buybacks': module1_buybacks,
            'module1_redistributions': module1_redistributions,
            'module1_protocol_inflow': module1_protocol_inflow,  # NEW
            'module1_burned': module1_burned,                    # NEW
            'module2_emissions': module2_emissions,
            'module3_emissions': module3_emissions,
            'module3_unlocked': module3_unlocked,
            
            # Locked tokens
            'staking_percent_vec': staking_percent_vec,
            'staked_tokens': staked_tokens,
            'module3_locked_tokens': module3_locked_tokens,
            'total_locked_tokens': total_locked_tokens,
            
            # State variables
            'module3_locked_balance_final': module3_locked_balance,
            'module3_milestones_reached': module3_unlock_reached,
            
            # Configuration snapshot
            'config': self.config,
            'price_model': self.price_trajectory.get_description()
        }
        
        return self.results
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Calculate summary metrics from simulation results
        
        Returns:
            Dictionary of key performance indicators
        """
        if not self.results:
            raise ValueError("Simulation must be run before calculating metrics")
        
        final_month = len(self.results['circulating_supply']) - 1
        
        return {
            'final_circulating_supply': self.results['circulating_supply'][final_month],
            'final_total_supply': self.results['total_supply'][final_month],
            'final_price': self.results['prices'][final_month],
            'final_fdv': self.results['fdv'][final_month],
            'average_monthly_net_flow': np.mean(self.results['net_flow']),
            'max_monthly_net_flow': np.max(self.results['net_flow']),
            'min_monthly_net_flow': np.min(self.results['net_flow']),
            'total_buybacks': np.sum(self.results['module1_buybacks']),
            'total_staking_emissions': np.sum(self.results['module2_emissions']),
            'total_participant_unlocked': np.sum(self.results['module3_unlocked']),
            'module3_still_locked': self.results['module3_locked_balance_final']
        }
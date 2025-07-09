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
    simulation_months: int = 120  # 10 years
    
    # Module 1 - Revenue Buybacks
    mrr_usd: float = 100_000  # Monthly recurring revenue in USD
    buyback_percentage: float = 0.80  # Percentage of revenue used for buybacks
    
    # Module 2 - DPoS Staking Rewards
    staking_budget_percentage: float = 0.15  # Percentage of total supply allocated to staking
    staking_half_life_months: float = 36  # Half-life for exponential decay in months
    
    # Staking percent (fraction of circulating supply staked)
    staking_percent: float = 0.3  # Default 30% staked
    
    # Module 3 - Participant Rewards
    participant_budget_percentage: float = 0.15  # Percentage of total supply for participant rewards
    participant_half_life_months: float = 48  # Half-life for exponential decay in months
    
    # Module 4 - Mining Reserve
    mining_reserve_percentage: float = 0.05  # Percentage of total supply for mining reserve
    mining_reserve_half_life_months: float = 60  # Half-life for exponential decay in months
    
    # Module 5 - Testnet Allocation
    testnet_allocation_percentage: float = 0.05  # Percentage of total supply for testnet allocation
    testnet_allocation_half_life_months: float = 72  # Half-life for exponential decay in months
    
    # FDV Milestones for Module 3 unlocks
    fdv_milestones: List[float] = None  # FDV thresholds for unlocking tokens
    unlock_percentages: List[float] = None  # Percentage of total budget unlocked at each milestone
    
    # Static Vesting Configuration
    static_vesting_data: Dict[str, List[int]] = None  # Custom static vesting data
    use_custom_static_vesting: bool = False  # Whether to use custom data or default
    
    # Static Vesting Extension Configuration
    extend_builder_rpgf: bool = True  # Whether to extend Builder and RPGF beyond Q20
    builder_rpgf_quarterly_growth: int = 1500000  # Quarterly growth rate for Builder/RPGF (1.5M)
    builder_rpgf_max_cap: int = 60000000  # Maximum cap for Builder/RPGF (60M)
    # max_quarters: int = 40  # REMOVED
    
    def __post_init__(self):
        """Set default milestone values if not provided and validate allocations"""
        if self.fdv_milestones is None:
            self.fdv_milestones = [100_000_000, 300_000_000, 1_000_000_000]  # $100M, $300M, $1B
        if self.unlock_percentages is None:
            self.unlock_percentages = [0.25, 0.50, 1.00]  # 25%, 50%, 100%
        
        # Validate custom static vesting data if provided
        if self.use_custom_static_vesting and self.static_vesting_data is not None:
            self.validate_static_vesting_data(self.static_vesting_data)
        
        # Validate that dynamic allocation buckets don't exceed 40%
        self._validate_dynamic_allocations()
    
    def _validate_dynamic_allocations(self):
        """Validate that the sum of dynamic allocation percentages doesn't exceed 40%"""
        dynamic_allocations = [
            self.staking_budget_percentage,
            self.participant_budget_percentage,
            self.mining_reserve_percentage,
            self.testnet_allocation_percentage
        ]
        
        total_dynamic_allocation = sum(dynamic_allocations)
        max_allowed = 0.40  # 40%
        
        if total_dynamic_allocation > max_allowed:
            raise ValueError(
                f"Total dynamic allocation ({total_dynamic_allocation:.1%}) exceeds maximum allowed ({max_allowed:.1%}). "
                f"Current allocations: Staking={self.staking_budget_percentage:.1%}, "
                f"Participant={self.participant_budget_percentage:.1%}, "
                f"MiningReserve={self.mining_reserve_percentage:.1%}, "
                f"TestnetAllocation={self.testnet_allocation_percentage:.1%}"
            )
    
    def get_dynamic_allocation_summary(self) -> Dict[str, float]:
        """Get a summary of all dynamic allocation percentages"""
        return {
            'staking_budget_percentage': self.staking_budget_percentage,
            'participant_budget_percentage': self.participant_budget_percentage,
            'mining_reserve_percentage': self.mining_reserve_percentage,
            'testnet_allocation_percentage': self.testnet_allocation_percentage,
            'total_dynamic_allocation': sum([
                self.staking_budget_percentage,
                self.participant_budget_percentage,
                self.mining_reserve_percentage,
                self.testnet_allocation_percentage
            ]),
            'remaining_for_static': 1.0 - sum([
                self.staking_budget_percentage,
                self.participant_budget_percentage,
                self.mining_reserve_percentage,
                self.testnet_allocation_percentage
            ])
        }
    
    @staticmethod
    def get_default_static_vesting_data() -> Dict[str, List[int]]:
        """Get the default static vesting data structure (base data up to Q20)"""
        return {
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
    
    @staticmethod
    def get_extended_static_vesting_data(extend_builder_rpgf: bool = True, 
                                        builder_rpgf_quarterly_growth: int = 1500000,
                                        builder_rpgf_max_cap: int = 60000000,
                                        simulation_months: int = 60) -> Dict[str, List[int]]:
        """
        Get extended static vesting data with configurable Builder/RPGF extension
        Args:
            extend_builder_rpgf: Whether to extend Builder and RPGF beyond Q20
            builder_rpgf_quarterly_growth: Quarterly growth rate for Builder/RPGF
            builder_rpgf_max_cap: Maximum cap for Builder/RPGF
            simulation_months: Number of months in the simulation (for alignment)
        Returns:
            Dictionary containing extended static vesting data
        """
        base_data = SimulationConfig.get_default_static_vesting_data()
        # Determine how many quarters are needed to reach the cap
        if extend_builder_rpgf:
            last_value = base_data['Builder'][-1]
            quarters_needed = int(np.ceil((builder_rpgf_max_cap - last_value) / builder_rpgf_quarterly_growth))
            max_quarter = 20 + quarters_needed
        else:
            max_quarter = len(base_data['Quarter']) - 1
        # Optionally, align with simulation duration (in months)
        max_quarter = max(max_quarter, (simulation_months - 1) // 3)
        quarters = list(range(max_quarter + 1))
        data = {'Quarter': quarters}
        # Extend each bucket
        for bucket in ['Titan_Labs', 'Titan_Foundation', 'Seed_Fundraising', 'SeriesA_Fundraising', 'Ecosystem', 'Testnet', 'Market_Making']:
            last_value = base_data[bucket][-1]
            if extend_builder_rpgf:
                data[bucket] = base_data[bucket] + [last_value] * (max_quarter - 20)
            else:
                data[bucket] = base_data[bucket]
        # Handle Builder and RPGF
        for bucket in ['Builder', 'RPGF']:
            if extend_builder_rpgf:
                extended_values = base_data[bucket].copy()
                current_value = base_data[bucket][-1]
                for quarter in range(21, max_quarter + 1):
                    if current_value < builder_rpgf_max_cap:
                        current_value = min(current_value + builder_rpgf_quarterly_growth, builder_rpgf_max_cap)
                    # After cap is reached, just keep the cap value
                    extended_values.append(current_value)
                data[bucket] = extended_values
            else:
                data[bucket] = base_data[bucket]
        return data
    
    @staticmethod
    def validate_static_vesting_data(data: Dict[str, List[int]]) -> bool:
        """
        Validate custom static vesting data structure
        
        Args:
            data: Dictionary containing static vesting data
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_keys = ['Quarter', 'Titan_Labs', 'Titan_Foundation', 'Seed_Fundraising', 
                        'SeriesA_Fundraising', 'Ecosystem', 'Testnet', 'Market_Making', 'Builder', 'RPGF']
        
        # Check all required keys exist
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Check all lists have the same length
        lengths = [len(data[key]) for key in required_keys]
        if len(set(lengths)) != 1:
            raise ValueError(f"All vesting schedules must have the same length. Found lengths: {lengths}")
        
        # Check quarters start from 0 and are sequential
        quarters = data['Quarter']
        if quarters[0] != 0:
            raise ValueError("Quarters must start from 0")
        
        for i in range(1, len(quarters)):
            if quarters[i] != quarters[i-1] + 1:
                raise ValueError(f"Quarters must be sequential. Found gap at index {i}")
        
        # Check all values are non-negative
        for key in required_keys:
            if key != 'Quarter':
                if any(val < 0 for val in data[key]):
                    raise ValueError(f"All values in {key} must be non-negative")
        
        # Check values are monotonically increasing (cumulative)
        for key in required_keys:
            if key != 'Quarter':
                for i in range(1, len(data[key])):
                    if data[key][i] < data[key][i-1]:
                        raise ValueError(f"Values in {key} must be monotonically increasing (cumulative)")
        
        return True


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
        
        # Use custom static vesting data if provided, otherwise use default
        if self.config.use_custom_static_vesting and self.config.static_vesting_data is not None:
            # If custom data is provided, use it directly without further extension
            # The custom data should already be properly extended if needed
            return pd.DataFrame(self.config.static_vesting_data)
        else:
            base_data = self.config.get_default_static_vesting_data()
        
        # Static vesting data provided (quarterly, cumulative)
        # Total Static Vesting = 1,140,000,000
        # Total Supply: 2,000,000,000
        # Total Static Percentage: 57%
        
        # Calculate how many quarters we need based on configuration
        if self.config.extend_builder_rpgf:
            # Calculate quarters needed to reach the cap, similar to get_extended_static_vesting_data
            last_value = base_data['Builder'][-1]  # 30,000,000 at Q20
            quarters_needed = int(np.ceil((self.config.builder_rpgf_max_cap - last_value) / self.config.builder_rpgf_quarterly_growth))
            max_quarter = 20 + quarters_needed
        else:
            # If not extending, use the original data length
            max_quarter = len(base_data['Quarter']) - 1
        
        # Extend quarters list
        quarters = list(range(max_quarter + 1))
        
        # Initialize extended data
        data = {'Quarter': quarters}
        
        # Extend each bucket to ensure all have the same length
        for bucket in ['Titan_Labs', 'Titan_Foundation', 'Seed_Fundraising', 'SeriesA_Fundraising', 'Ecosystem', 'Testnet', 'Market_Making']:
            # These buckets are already at their maximum values, so extend with the last value
            last_value = base_data[bucket][-1]
            if max_quarter > 20:  # Only extend if we need more quarters than the base data
                data[bucket] = base_data[bucket] + [last_value] * (max_quarter - 20)
            else:
                data[bucket] = base_data[bucket]
        
        # Handle Builder and RPGF based on configuration
        for bucket in ['Builder', 'RPGF']:
            if self.config.extend_builder_rpgf:
                # Extend Builder and RPGF with continued growth
                extended_values = base_data[bucket].copy()
                current_value = base_data[bucket][-1]  # 30,000,000 at Q20
                
                for quarter in range(21, max_quarter + 1):
                    current_value += self.config.builder_rpgf_quarterly_growth
                    if current_value > self.config.builder_rpgf_max_cap:  # Cap at configured maximum
                        current_value = self.config.builder_rpgf_max_cap
                    extended_values.append(current_value)
                
                data[bucket] = extended_values
            else:
                # Use original data, but extend with last value if needed to match length
                if max_quarter > 20:
                    data[bucket] = base_data[bucket] + [base_data[bucket][-1]] * (max_quarter - 20)
                else:
                    data[bucket] = base_data[bucket]
        
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
        mining_reserve_balance = np.zeros(months)  # NEW: Cumulative Mining Reserve balance
        testnet_allocation_emissions = np.zeros(months)  # NEW: Testnet Allocation emissions
        
        # New: Staking percent vector and locked tokens tracking
        staking_percent_vec = np.full(months, self.config.staking_percent)
        staked_tokens = np.zeros(months)
        module3_locked_tokens = np.zeros(months)
        total_locked_tokens = np.zeros(months)
        
        # NEW: Buyback/burn constraint tracking
        buyback_constraint_violations = np.zeros(months, dtype=bool)
        burn_constraint_violations = np.zeros(months, dtype=bool)
        max_buyable_tokens = np.zeros(months)
        max_burnable_tokens = np.zeros(months)
        
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
        mining_reserve_total_budget = self.config.total_supply * self.config.mining_reserve_percentage
        testnet_allocation_total_budget = self.config.total_supply * self.config.testnet_allocation_percentage
        
        # Main simulation loop - process each month
        for month in range(months):
            # Get current token price
            prices[month] = self.price_trajectory.get_price(month)
            
            # Calculate total supply (static vesting + unlocked dynamic emissions)
            # Note: mining_reserve is excluded as it's permanently locked
            total_supply_over_time[month] = (
                total_static_monthly[month] +
                np.sum(module2_emissions[:month+1]) +
                np.sum(module3_unlocked[:month+1]) +
                np.sum(testnet_allocation_emissions[:month+1])
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
            # Redistribute tokens to workers (80% of bought tokens)
            module1_redistributions[month] = tokens_bought * 0.80
            
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
            
            # Track cumulative mining reserve balance (constant after month 0)
            mining_reserve_balance[month] = mining_reserve_total_budget
            
            # Testnet Allocation (exponential decay, direct to circulation)
            testnet_allocation_emissions[month] = self._calculate_exponential_decay_emission(
                month, testnet_allocation_total_budget, self.config.testnet_allocation_half_life_months
            )
            
            # Calculate circulating supply and check constraints as we build it
            if month == 0:
                circulating_supply[month] = static_vesting_flow[month]
            else:
                circulating_supply[month] = circulating_supply[month-1] + static_vesting_flow[month]
            
            # Add dynamic emissions to circulation
            circulating_supply[month] += module2_emissions[month]  # Staking rewards
            circulating_supply[month] += module3_unlocked[month]  # Unlocked participant rewards
            circulating_supply[month] += testnet_allocation_emissions[month]  # Testnet Allocation (direct to circulation)
            # Module1 flows: tokens are bought from circulation, then redistributed back
            # Net effect: only burned tokens are permanently removed from circulation
            # Subtract burned tokens (permanently removed)
            circulating_supply[month] -= module1_burned[month]
            # Note: redistributed tokens (80%) and protocol tokens (10%) remain in circulation
            # since they were bought from circulation and redistributed back
            
            # Check buyback/burn constraints against available circulating supply BEFORE module1 flows
            available_circulating = circulating_supply[month]
            max_buyable_tokens[month] = available_circulating
            max_burnable_tokens[month] = available_circulating
            
            buyback_constraint_violations[month] = tokens_bought > available_circulating
            burn_constraint_violations[month] = tokens_burned > available_circulating
            
            # Calculate staked tokens for this month (after all emissions)
            staked_tokens[month] = circulating_supply[month] * staking_percent_vec[month]
            # Total locked = staked + module3 locked
            total_locked_tokens[month] = staked_tokens[month] + module3_locked_tokens[month]
        
        # Calculate net flow as derivative of circulating supply (positive = net emission, negative = net sink)
        net_flow = np.zeros(months)
        for month in range(months):
            if month == 0:
                net_flow[month] = circulating_supply[month]
            else:
                net_flow[month] = circulating_supply[month] - circulating_supply[month-1]
        
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
            'mining_reserve_balance': mining_reserve_balance,  # NEW: Cumulative Mining Reserve balance
            'testnet_allocation_emissions': testnet_allocation_emissions,  # NEW: Testnet Allocation
            
            # Locked tokens
            'staking_percent_vec': staking_percent_vec,
            'staked_tokens': staked_tokens,
            'module3_locked_tokens': module3_locked_tokens,
            'total_locked_tokens': total_locked_tokens,
            
            # NEW: Buyback/burn constraint tracking
            'buyback_constraint_violations': buyback_constraint_violations,
            'burn_constraint_violations': burn_constraint_violations,
            'max_buyable_tokens': max_buyable_tokens,
            'max_burnable_tokens': max_burnable_tokens,
            
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
            'total_mining_reserve_allocation': self.results['mining_reserve_balance'][final_month],
            'total_testnet_allocation_emissions': np.sum(self.results['testnet_allocation_emissions']),
            'module3_still_locked': self.results['module3_locked_balance_final']
        }
    
    def analyze_buyback_constraints(self) -> Dict[str, Any]:
        """
        Analyze buyback and burn constraint violations and calculate maximum sustainable revenue
        
        Returns:
            Dictionary containing constraint analysis results
        """
        if not self.results:
            raise ValueError("Simulation must be run before analyzing constraints")
        
        # Find months with constraint violations
        buyback_violation_months = np.where(self.results['buyback_constraint_violations'])[0]
        burn_violation_months = np.where(self.results['burn_constraint_violations'])[0]
        
        # Calculate maximum sustainable revenue for each month
        max_sustainable_revenue = []
        for month in range(len(self.results['months'])):
            if self.results['prices'][month] > 0:
                # Calculate how much USD revenue could be converted to tokens
                max_tokens = self.results['max_buyable_tokens'][month]
                max_revenue_for_buybacks = max_tokens * self.results['prices'][month] / self.config.buyback_percentage
                max_revenue_for_burns = max_tokens * self.results['prices'][month] / 0.10  # 10% of revenue
                # Take the minimum (most restrictive constraint)
                max_sustainable_revenue.append(min(max_revenue_for_buybacks, max_revenue_for_burns))
            else:
                max_sustainable_revenue.append(0)
        
        max_sustainable_revenue = np.array(max_sustainable_revenue)
        
        # Find the minimum sustainable revenue (bottleneck)
        min_sustainable_revenue = np.min(max_sustainable_revenue[max_sustainable_revenue > 0]) if np.any(max_sustainable_revenue > 0) else 0
        
        # NEW: Buy pressure analysis
        buy_pressure_metrics = self._calculate_buy_pressure_metrics()
        
        return {
            'buyback_violation_months': buyback_violation_months.tolist(),
            'burn_violation_months': burn_violation_months.tolist(),
            'total_buyback_violations': len(buyback_violation_months),
            'total_burn_violations': len(burn_violation_months),
            'max_sustainable_revenue': max_sustainable_revenue,
            'min_sustainable_revenue': min_sustainable_revenue,
            'current_revenue': self.config.mrr_usd,
            'revenue_constrained': self.config.mrr_usd > min_sustainable_revenue if min_sustainable_revenue > 0 else False,
            'constraint_bottleneck_month': np.argmin(max_sustainable_revenue) if np.any(max_sustainable_revenue > 0) else None,
            'buy_pressure_metrics': buy_pressure_metrics
        }
    
    def _calculate_buy_pressure_metrics(self) -> Dict[str, Any]:
        """
        Calculate buy pressure metrics to understand MRR's impact on token demand
        
        Returns:
            Dictionary containing buy pressure analysis
        """
        if not self.results:
            raise ValueError("Simulation must be run before calculating buy pressure metrics")
        
        months = len(self.results['months'])
        
        # Calculate monthly buy pressure metrics
        monthly_buy_pressure = []
        monthly_buyback_volume_usd = []
        monthly_buyback_volume_tokens = []
        monthly_buyback_percentage_of_supply = []
        monthly_buyback_percentage_of_flow = []
        
        for month in range(months):
            # Buyback volume in USD
            buyback_usd = self.config.mrr_usd * self.config.buyback_percentage
            monthly_buyback_volume_usd.append(buyback_usd)
            
            # Buyback volume in tokens
            buyback_tokens = self.results['module1_buybacks'][month]
            monthly_buyback_volume_tokens.append(buyback_tokens)
            
            # Buyback as percentage of circulating supply
            circulating_supply = self.results['circulating_supply'][month]
            buyback_pct_supply = (buyback_tokens / circulating_supply * 100) if circulating_supply > 0 else 0
            monthly_buyback_percentage_of_supply.append(buyback_pct_supply)
            
            # Buyback as percentage of monthly net flow
            net_flow = self.results['net_flow'][month]
            buyback_pct_flow = (buyback_tokens / abs(net_flow) * 100) if abs(net_flow) > 0 else 0
            monthly_buyback_percentage_of_flow.append(buyback_pct_flow)
            
            # Buy pressure score (combines volume and supply impact)
            buy_pressure_score = buyback_pct_supply * (buyback_usd / 1000)  # Normalized by revenue scale
            monthly_buy_pressure.append(buy_pressure_score)
        
        # Calculate summary statistics
        avg_buyback_pct_supply = np.mean(monthly_buyback_percentage_of_supply)
        max_buyback_pct_supply = np.max(monthly_buyback_percentage_of_supply)
        avg_buy_pressure_score = np.mean(monthly_buy_pressure)
        max_buy_pressure_score = np.max(monthly_buy_pressure)
        
        # Calculate total buyback impact over simulation
        total_buyback_usd = np.sum(monthly_buyback_volume_usd)
        total_buyback_tokens = np.sum(monthly_buyback_volume_tokens)
        
        # Calculate buyback efficiency (tokens bought per USD)
        avg_buyback_efficiency = total_buyback_tokens / total_buyback_usd if total_buyback_usd > 0 else 0
        
        # Find peak buy pressure month
        peak_buy_pressure_month = np.argmax(monthly_buy_pressure)
        
        return {
            'monthly_buy_pressure': np.array(monthly_buy_pressure),
            'monthly_buyback_volume_usd': np.array(monthly_buyback_volume_usd),
            'monthly_buyback_volume_tokens': np.array(monthly_buyback_volume_tokens),
            'monthly_buyback_percentage_of_supply': np.array(monthly_buyback_percentage_of_supply),
            'monthly_buyback_percentage_of_flow': np.array(monthly_buyback_percentage_of_flow),
            'avg_buyback_pct_supply': avg_buyback_pct_supply,
            'max_buyback_pct_supply': max_buyback_pct_supply,
            'avg_buy_pressure_score': avg_buy_pressure_score,
            'max_buy_pressure_score': max_buy_pressure_score,
            'total_buyback_usd': total_buyback_usd,
            'total_buyback_tokens': total_buyback_tokens,
            'avg_buyback_efficiency': avg_buyback_efficiency,
            'peak_buy_pressure_month': peak_buy_pressure_month
        }
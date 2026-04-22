# Fixed Deposit Calculator
# This module calculates FD returns using simple formulas

def calculate_fd_return(principal, rate, time_period):
    """
    Calculate fixed deposit return using compound interest
    
    Args:
        principal (float): Initial investment amount
        rate (float): Annual interest rate (in percentage)
        time_period (int): Time period in years
        
    Returns:
        dict: Contains total amount, interest earned, and risk level
    """
    # Convert rate from percentage to decimal
    rate_decimal = rate / 100
    
    # Calculate compound interest: A = P(1 + r)^t
    total_amount = principal * (1 + rate_decimal) ** time_period
    interest_earned = total_amount - principal
    
    return {
        'total_amount': total_amount,
        'interest_earned': interest_earned,
        'principal': principal,
        'rate': rate,
        'time_period': time_period
    }

def get_fd_risk_level():
    """
    Get risk level for fixed deposit (always Low)
    
    Returns:
        str: Risk level
    """
    return "Low"

def get_fd_rate(time_period):
    """
    Get FD interest rate based on time period
    
    Args:
        time_period (int): Time period in years
        
    Returns:
        float: Interest rate percentage
    """
    # Define FD rates based on time period (typical bank rates)
    if time_period <= 1:
        return 5.5  # 5.5% for 1 year or less
    elif time_period <= 3:
        return 6.5  # 6.5% for 2-3 years
    elif time_period <= 5:
        return 7.0  # 7.0% for 4-5 years
    else:
        return 7.5  # 7.5% for more than 5 years

def analyze_fd_investment(amount, time_period):
    """
    Analyze fixed deposit investment
    
    Args:
        amount (float): Investment amount
        time_period (int): Time period in years
        
    Returns:
        dict: Analysis results
    """
    # Get appropriate FD rate
    rate = get_fd_rate(time_period)
    
    # Calculate returns
    result = calculate_fd_return(amount, rate, time_period)
    
    # Add risk level and explanation
    result['risk_level'] = get_fd_risk_level()
    result['explanation'] = "Low risk guaranteed returns with fixed deposit"
    
    return result

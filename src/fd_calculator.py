"""
Fixed Deposit Calculator — simple math formulas, no ML model needed.
Moved from src/fd/fd_calculator.py — no subfolder needed for a single file.
"""


def calculate_fd_return(principal, rate, time_period):
    """
    Calculate Fixed Deposit return using compound interest.

    Args:
        principal (float): Initial investment amount
        rate (float): Annual interest rate as a percentage (e.g. 7.5 means 7.5%)
        time_period (int): Time period in years

    Returns:
        dict: total_amount, interest_earned, principal, rate, time_period
    """
    rate_decimal = rate / 100
    total_amount = principal * (1 + rate_decimal) ** time_period
    interest_earned = total_amount - principal

    return {
        'total_amount': total_amount,
        'interest_earned': interest_earned,
        'principal': principal,
        'rate': rate,
        'time_period': time_period,
    }


def get_fd_rate(time_period):
    """
    Get a typical FD interest rate based on the investment time period.

    Args:
        time_period (int): Time period in years

    Returns:
        float: Interest rate percentage
    """
    if time_period <= 1:
        return 5.5   # 5.5% for 1 year or less
    elif time_period <= 3:
        return 6.5   # 6.5% for 2-3 years
    elif time_period <= 5:
        return 7.0   # 7.0% for 4-5 years
    else:
        return 7.5   # 7.5% for more than 5 years


def get_fd_risk_level():
    """
    FD risk level is always Low — guaranteed government-backed returns.

    Returns:
        str: 'Low'
    """
    return "Low"


def analyze_fd_investment(amount, time_period):
    """
    Full FD investment analysis — rate lookup, return calculation, risk tagging.

    Args:
        amount (float): Investment amount
        time_period (int): Time period in years

    Returns:
        dict: Analysis results including total_amount, interest_earned, risk_level, explanation
    """
    rate = get_fd_rate(time_period)
    result = calculate_fd_return(amount, rate, time_period)
    result['risk_level'] = get_fd_risk_level()
    result['explanation'] = "Low risk - guaranteed returns with fixed deposit"
    return result
"""Two calculator tools for the agent.

The LLM reads the numbers from the policy and passes them here,
so dates and money are always calculated correctly.
"""

import calendar
from datetime import date
from typing import Optional

from langchain_core.tools import tool


def add_months(start_date: date, months: int) -> date:
    """Add months to a date. If the day doesn't exist (31 Feb), use the last day of that month."""
    month_index = start_date.month - 1 + months
    year = start_date.year + month_index // 12
    month = month_index % 12 + 1
    last_day_of_month = calendar.monthrange(year, month)[1]
    return date(year, month, min(start_date.day, last_day_of_month))


@tool
def check_waiting_period(cover_start_date: str, waiting_period_months: int, treatment_date: str) -> str:
    """Check if a waiting period is over on the treatment date.
    cover_start_date: date the FIRST policy started, YYYY-MM-DD (read it from the policy).
    waiting_period_months: from the policy, e.g. 24 for cataract, 36 for pre-existing diseases.
    treatment_date: date of the treatment, YYYY-MM-DD."""
    start = date.fromisoformat(cover_start_date)
    treatment = date.fromisoformat(treatment_date)
    waiting_ends = add_months(start, waiting_period_months)

    if treatment >= waiting_ends:
        return (f"Cover started on {start}. The {waiting_period_months}-month waiting period ended on {waiting_ends}. "
                f"Treatment on {treatment} is after that, so the waiting period is over.")
    return (f"Cover started on {start}. The {waiting_period_months}-month waiting period ends on {waiting_ends}. "
            f"Treatment on {treatment} is before that, so the waiting period is not over yet.")


@tool
def estimate_claim(
    total_bill: float,
    medicines_and_tests: float,
    room_rent_per_day: float,
    sum_insured: float,
    copay_percent: float,
    room_rent_limit_per_day: Optional[float] = None,
    room_rent_limit_percent: Optional[float] = None,
) -> str:
    """Estimate how much the insurer pays for a hospital bill (amounts in rupees).
    The number of days in hospital is not needed.
    total_bill: the full hospital bill.
    medicines_and_tests: the part of the bill for medicines, consumables, implants and tests
        (these are NOT cut when the room is costlier than allowed). Use 0 if not known.
    room_rent_per_day: what the hospital charged for the room per day.
    room_rent_limit_per_day: the policy's room limit in rupees per day.
    room_rent_limit_percent: the limit as % of sum insured per day (e.g. 2 for 2%).
        If the policy says "2% of sum insured, maximum ₹5,000 per day", give BOTH.
    copay_percent: the policy's co-payment in % (e.g. 5). REQUIRED: first call search_my_policy("co-payment").
        Use 0 only if the policy has no co-payment."""
    # 0. Catch mixed-up numbers, so the agent can correct them
    if medicines_and_tests > total_bill:
        return "medicines_and_tests can't be more than total_bill. Check the amounts and call again."


    # 1. The room rent limit in rupees per day.
    #    If the policy gives a percent AND a rupee maximum, the lower one applies.
    if room_rent_limit_percent is not None:
        percent_limit = sum_insured * room_rent_limit_percent / 100
        if room_rent_limit_per_day is None or percent_limit < room_rent_limit_per_day:
            room_rent_limit_per_day = percent_limit

    # 2. Proportionate deduction: if the room was costlier than allowed,
    #    every charge except medicines and tests is paid only in the same ratio
    amount_covered = total_bill
    explanation = f"Total bill: ₹{total_bill:,.0f}. "

    if room_rent_limit_per_day and room_rent_per_day > room_rent_limit_per_day:
        ratio = room_rent_limit_per_day / room_rent_per_day
        other_charges = total_bill - medicines_and_tests
        amount_covered = other_charges * ratio + medicines_and_tests
        explanation += (f"Room ₹{room_rent_per_day:,.0f}/day is above the limit ₹{room_rent_limit_per_day:,.0f}/day, "
                        f"so all charges except medicines and tests are paid at {ratio:.1%}: "
                        f"₹{other_charges:,.0f} × {ratio:.1%} + ₹{medicines_and_tests:,.0f} = ₹{amount_covered:,.0f}. ")
    else:
        explanation += "Room rent is within the limit, so nothing is cut. "

     # 3. The insurer never pays more than the sum insured
    if amount_covered > sum_insured:
        amount_covered = sum_insured
        explanation += f"Capped at the sum insured: ₹{sum_insured:,.0f}. "

    # 4. Co-payment: the patient's own share, taken from the capped amount
    copay_amount = amount_covered * copay_percent / 100
    insurer_pays = amount_covered - copay_amount
    if copay_percent:
        explanation += f"Co-payment {copay_percent:g}%: −₹{copay_amount:,.0f}. "

    explanation += (f"Insurer pays about ₹{insurer_pays:,.0f}, you pay about ₹{total_bill - insurer_pays:,.0f}. "
                    "(Rough estimate: items the policy never pays for are not included.)")
    return explanation

import pymupdf
doc = pymupdf.open()
pages = [
"""POLICY SCHEDULE
Insured: Ravi Kumar  Mobile: 9876543210  Email: ravi@example.com  PAN: ABCDE1234F
Insurer: Demo General Insurance Ltd   Plan: Demo Health Secure
Policy Period: 01/04/2025 to 31/03/2026
Date of first inception (continuous cover since): 2023-04-01
Sum Insured: Rs. 5,00,000""",
"""4. Benefits
4.1 In-patient Hospitalisation
We will cover medical expenses for hospitalisation exceeding 24 hours.
4.2 Room Rent
Room rent is payable up to 1% of the Sum Insured per day. ICU charges up to 2% of Sum Insured per day.
If the insured person opts for a room with higher rent, all associated medical expenses
shall be reduced proportionately.
4.3 Co-payment
A co-payment of 10% applies to every admissible claim.""",
"""5. Waiting Periods
5.1 Initial Waiting Period
Expenses within 30 days from the first policy commencement are excluded, except accidents.
5.2 Specific Waiting Period
Cataract, hernia, joint replacement and benign prostatic hypertrophy are covered after 24 months
of continuous coverage.
5.3 Pre-existing Diseases
Pre-existing diseases are covered after 36 months of continuous coverage.
6. Exclusions
Excl01 Cosmetic surgery is excluded unless required due to an accident.""",
]
for t in pages:
    p = doc.new_page()
    p.insert_text((50, 60), t, fontsize=10)
doc.save("tests/demo_policy.pdf")

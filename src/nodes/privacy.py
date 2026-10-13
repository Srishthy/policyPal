"""Hide personal details before anything is sent to the LLM.

We use LangChain's PIIMiddleware. It runs before every LLM call and checks:
  - the user's message (which also contains the policy clauses and the chat history)
  - every tool result (policy search, IRDAI search, web search)
Anything that matches is replaced with a placeholder like [REDACTED_PHONE].
"""

from langchain.agents.middleware import PIIMiddleware

# Aadhaar: 12 digits, often written as 1234 5678 9012
AADHAAR_PATTERN = r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"

# PAN: 5 letters, 4 digits, 1 letter, like ABCDE1234F
PAN_PATTERN = r"\b[A-Z]{5}\d{4}[A-Z]\b"

# Indian mobile number: 10 digits starting with 6, 7, 8 or 9.
# Also catches +91 / 91 / 0 in front and a space after 5 digits: +919876543210, 98765 43210, 09876543210
PHONE_PATTERN = r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"


def pii_middleware() -> list:
    """One middleware per type of personal detail. Aadhaar comes before phone,
    so a 12-digit Aadhaar number isn't partly mistaken for a phone number."""
    settings = {"strategy": "redact", "apply_to_input": True, "apply_to_tool_results": True}
    return [
        PIIMiddleware("aadhaar", detector=AADHAAR_PATTERN, **settings),
        PIIMiddleware("pan", detector=PAN_PATTERN, **settings),
        PIIMiddleware("phone", detector=PHONE_PATTERN, **settings),
        PIIMiddleware("email", **settings),  # email detection is built into LangChain
    ]

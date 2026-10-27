import re

from langsmith import traceable

from guardrails import Guard, OnFailAction
from guardrails.errors import ValidationError
from guardrails.validators import FailResult, PassResult, Validator, register_validator


# ------------------------------------------------------------------
# SETTINGS (edit these lists to suit your project)
# ------------------------------------------------------------------
MAX_QUESTION_LENGTH = 500

# phrases people use to try to take over the model (compared in lowercase)
INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "ignore your instructions",
    "disregard previous",
    "disregard the above",
    "forget your instructions",
    "forget all previous",
    "reveal your system prompt",
    "show your system prompt",
    "print your system prompt",
    "what is your system prompt",
    "developer mode",
    "jailbreak",
    "you are now",
    "pretend you are",
    "openai key",
    "openai-key",
    "api key",
    "api-key",
    "password",
]

# text from our own system prompts that must never appear in an answer
SYSTEM_PROMPT_SNIPPETS = [
    "answer the question using only the context below",
    "classify the user message into exactly one word",
    "rewrite the user's latest message as a standalone question",
]

# personal data and secrets: name -> pattern
PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "indian mobile": r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
    "international phone": r"\+\d{1,3}[\s-]?\d{6,12}(?!\d)",
    "card number": r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
    "openai key": r"sk-[A-Za-z0-9_-]{20,}",
}


# ------------------------------------------------------------------
# PLAIN HELPER FUNCTIONS (no Guardrails in here, easy to read and test)
# ------------------------------------------------------------------
def find_injection(text):
    """Return the first injection phrase found in the text, or None."""
    lowered = " ".join(text.lower().split())      # lowercase + squeeze whitespace
    for phrase in INJECTION_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def find_pii(text):
    """Return the names of all PII/secret types found in the text."""
    return [name for name, pattern in PII_PATTERNS.items() if re.search(pattern, text)]


def redact(text):
    """Replace every PII/secret match with [REDACTED]."""
    for pattern in PII_PATTERNS.values():
        text = re.sub(pattern, "[REDACTED]", text)
    return text


# ------------------------------------------------------------------
# GUARDRAILS VALIDATORS (thin wrappers around the helpers above)
# A validator returns PassResult() or FailResult(error_message=...)
# ------------------------------------------------------------------
@register_validator(name="rag/max_length", data_type="string")
class MaxLength(Validator):
    def _validate(self, value, metadata):
        if len(value) > MAX_QUESTION_LENGTH:
            return FailResult(
                error_message=f"Your question is too long (maximum {MAX_QUESTION_LENGTH} characters).")
        return PassResult()


@register_validator(name="rag/no_injection", data_type="string")
class NoInjection(Validator):
    def _validate(self, value, metadata):
        if find_injection(value):
            return FailResult(
                error_message="That looks like an attempt to change my instructions, so I can't process it.")
        return PassResult()


@register_validator(name="rag/no_pii_block", data_type="string")
class NoPIIBlock(Validator):
    """For INPUT: refuse the question if it contains personal data or secrets."""
    def _validate(self, value, metadata):
        if find_pii(value):
            return FailResult(
                error_message="Please don't include personal data or secrets in your question.")
        return PassResult()


@register_validator(name="rag/no_pii_redact", data_type="string")
class NoPIIRedact(Validator):
    """For OUTPUT: hide personal data and secrets instead of refusing the whole answer."""
    def _validate(self, value, metadata):
        if find_pii(value):
            return FailResult(error_message="Answer contained personal data.",
                              fix_value=redact(value))
        return PassResult()


@register_validator(name="rag/no_prompt_leak", data_type="string")
class NoPromptLeak(Validator):
    def _validate(self, value, metadata):
        lowered = " ".join(value.lower().split())
        if any(snippet in lowered for snippet in SYSTEM_PROMPT_SNIPPETS):
            return FailResult(error_message="Answer contained internal instructions.")
        return PassResult()


# ------------------------------------------------------------------
# THE GUARDS (a guard = a list of validators that run in order)
# EXCEPTION = stop and raise an error, FIX = replace with the fixed value
# ------------------------------------------------------------------
input_guard = (
    Guard(name="input_guard")
    .use(MaxLength(on_fail=OnFailAction.EXCEPTION))
    .use(NoInjection(on_fail=OnFailAction.EXCEPTION))
    .use(NoPIIBlock(on_fail=OnFailAction.EXCEPTION))
)

output_guard = (
    Guard(name="output_guard")
    .use(NoPromptLeak(on_fail=OnFailAction.EXCEPTION))
    .use(NoPIIRedact(on_fail=OnFailAction.FIX))
)


# ------------------------------------------------------------------
# THE THREE FUNCTIONS YOU CALL FROM rag.py
# ------------------------------------------------------------------
@traceable(name = "check_input")
def check_input(question):
    """Returns (True, "") if question is safe, else (False, reason)."""
    # 1. Check Length
    if len(question) > MAX_QUESTION_LENGTH:
        return False, f"Your question is too long (maximum {MAX_QUESTION_LENGTH} characters)."

    # 2. Check Prompt Injection (matches any phrase regardless of formatting)
    found_phrase = find_injection(question)
    if found_phrase:
        print(f"[GUARDRAIL TRIGGERED] Injection phrase caught: '{found_phrase}'")
        return False, "That looks like an attempt to change my instructions, so I can't process it."

    # 3. Check PII / Secrets
    pii_types = find_pii(question)
    if pii_types:
        print(f"[GUARDRAIL TRIGGERED] PII/Secret caught: {pii_types}")
        return False, "Please don't include personal data or secrets in your question."

    return True, ""


@traceable(name = "clean_output")
def clean_output(answer):
    """Returns the answer, with personal data hidden, or a safe message if it must be blocked."""
    try:
        return output_guard.validate(answer).validated_output
    except ValidationError:
        return "Sorry, I can't share that."

@traceable(name = "chunk_is_safe")
def chunk_is_safe(text):
    """False if a retrieved chunk tries to give the model instructions."""
    return find_injection(text) is None
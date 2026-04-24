from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import logging
logger = logging.getLogger(__name__)

def log_attempt_failed(retry_state):
    logger.warning(
        f"Attempt {retry_state.attempt_number} failed. "
        f"Retrying in {retry_state.idle_for} seconds..."
    )

def circuit_breaker(max_attempts=3, min_wait=1, max_wait=10):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        after=log_attempt_failed
    )

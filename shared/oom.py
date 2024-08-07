import torch
import functools
import logging


def with_oom_protection(max_retries=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and attempt < max_retries:
                        logging.info(
                            f"""🛡️ 🔴 CUDA out of memory error encountered in "{func.__name__}". Clearing cache and retrying (attempt {attempt + 1}/{max_retries})..."""
                        )
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    else:
                        logging.info(f'🛡️ 🔴 Error in "{func.__name__}": {e}')
                        raise  # Re-raise the exception if it's not OOM or we've exceeded retries

            err = f"""🛡️ 🔴 Function "{func.__name__}" failed after {max_retries} retries due to CUDA out of memory errors."""
            logging.info(err)
            raise RuntimeError(err)

        return wrapper

    return decorator

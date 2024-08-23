import torch
import functools
import logging
import inspect


from shared.move_to_cpu import move_all_models_to_cpu


def with_oom_protection(max_retries=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and attempt < max_retries:
                        logging.info(
                            f"""🛡️ 🔴 CUDA out of memory error encountered in "{func.__name__}". Clearing cache and retrying (attempt {attempt + 1}/{max_retries})..."""
                        )
                        # Move all models to cpu to start fresh
                        try:
                            logging.info("🛡️ Trying to move all models to CPU after OOM")
                            models_pack = kwargs.get("models_pack", None)
                            if models_pack is not None:
                                logging.info(f"🛡️ 🟣 Models pack found in kwargs")
                            else:
                                logging.info(f"🛡️ 🟠 Models pack not found in kwargs")
                            if models_pack is None and "models_pack" in param_names:
                                index = param_names.index("models_pack")
                                if index < len(args):
                                    models_pack = args[index]
                                    logging.info(f"🛡️ 🟣 Models pack found in args")
                                else:
                                    logging.info(f"🛡️ 🟠 Models pack not found in args")
                            if models_pack:
                                logging.info(f"🛡️ 🟡 Moving models_pack to cpu")
                                move_all_models_to_cpu(models_pack)
                                logging.info(f"🛡️ 🟢 Moved models_pack to cpu")
                            else:
                                logging.info(
                                    f"🛡️ 🔴 Models_pack not found in kwargs or args"
                                )
                        except Exception as e:
                            logging.info(
                                f'🛡️ 🔴 Error moving models to CPU in "{func.__name__}": {e}'
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

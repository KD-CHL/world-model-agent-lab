"""Terminal-only training feedback; never part of the optimizer or model metadata."""
import sys


class _DisabledProgressBar:
    """Drop-in context for training callers when progress output is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_postfix(self, *args, **kwargs):
        return None

    def update(self, n=1):
        return None


def progress_bar(total, description, *, enabled=False, unit='batch', initial=0, position=0, leave=True):
    if not enabled:
        return _DisabledProgressBar()
    from tqdm import tqdm
    # Cursor-up nesting is useful in a terminal, but corrupts redirected plain logs.
    if not sys.stderr.isatty():
        position=0
    return tqdm(total=total,initial=initial,desc=description,unit=unit,
                file=sys.stderr,dynamic_ncols=True,ascii=True,
                mininterval=.25,position=position,leave=leave,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '[{elapsed}, ETA {remaining}]{postfix}')


def training_status(message, *, enabled=False):
    if enabled:
        print(message,file=sys.stderr,flush=True)


def epoch_status(epoch,total,train_loss,val_loss,best,*,enabled=False):
    if enabled:
        from tqdm import tqdm
        # A persistent line retains all metrics even when the terminal bar is narrow.
        tqdm.write(f'Epoch {epoch}/{total}: train_loss={train_loss:.6g} '
                   f'val_loss={val_loss:.6g} best={best:.6g}',file=sys.stderr)

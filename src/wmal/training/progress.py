"""Terminal-only training feedback; never part of the optimizer or model metadata."""
import sys


def progress_bar(total, description, *, enabled=False, unit='batch', initial=0, position=0, leave=True):
    from tqdm import tqdm
    # Cursor-up nesting is useful in a terminal, but corrupts redirected plain logs.
    if not sys.stderr.isatty():
        position=0
    return tqdm(total=total,initial=initial,desc=description,unit=unit,
                disable=not enabled,file=sys.stderr,dynamic_ncols=True,ascii=True,
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

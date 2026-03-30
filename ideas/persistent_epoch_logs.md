# Idea: Persistent Epoch Log Messages

## Problem
`livelossplot` calls `clear_output()` each time it redraws the plot, which wipes all previous text output in the notebook cell. Epoch summaries printed via `tqdm.write()` or `print()` disappear.

## Solution
Accumulate epoch messages in a list and reprint them all after each `liveloss.send()` call.

### Changes to `src/train.py` in `optimize()`:

1. Add `log_messages = []` after `logs = {}`

2. Replace the `tqdm.write()` calls with message accumulation:

```python
# print training/validation statistics
msg = "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
        epoch, train_loss, valid_loss
)

# If the validation loss decreases by more than 1%, save the model
if valid_loss_min is None or (
        (valid_loss_min - valid_loss) / valid_loss_min > 0.01
):
    msg += f"\n  New minimum validation loss: {valid_loss:.6f}. Saving model ..."
    torch.save(model.state_dict(), save_path)
    valid_loss_min = valid_loss

log_messages.append(msg)
```

3. After `liveloss.send()`, reprint all messages:

```python
if interactive_tracking:
    logs["loss"] = train_loss
    logs["val_loss"] = valid_loss
    logs["lr"] = optimizer.param_groups[0]["lr"]

    liveloss.update(logs)
    liveloss.send()

    # Reprint all epoch summaries after plot refresh
    print("\n".join(log_messages))
else:
    print(msg)
```

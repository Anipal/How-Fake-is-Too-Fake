from tqdm import tqdm

def progress_bar(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
    """
    Display a progress bar with aggregated statistics for each epoch.

    Args:
    - epoch (int): Current epoch.
    - total_epochs (int): Total number of epochs.
    - train_loss (float): Training loss for the epoch.
    - val_loss (float): Validation loss for the epoch.
    - train_correct (int): Number of correct predictions for training data for the epoch.
    - train_total (int): Total number of samples for training data for the epoch.
    - val_correct (int): Number of correct predictions for validation data for the epoch.
    - val_total (int): Total number of samples for validation data for the epoch.
    """
    # Create the progress bar for the epoch
    progress = tqdm(total=1, position=0, leave=True, desc=f'Epoch {epoch}/{total_epochs}')
    # Update the progress bar with the aggregated statistics for the epoch
    progress.set_postfix_str(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% '
                             f'| Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}%')
    # Close the progress bar when finished
    progress.close()
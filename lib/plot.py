
def plot_training_stat(stats, key, axes):
    axes.plot(stats['epoch'],stats[key])
    axes.plot(stats['epoch'],stats['val_' + key])
    axes.set_ylabel(key)
    axes.set_xlabel('Epoch')
    axes.set_title(key)
    axes.legend(['train' , 'validation'] , loc = 'upper right')
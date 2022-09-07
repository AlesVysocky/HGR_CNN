import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.ticker as mtick

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(paths):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }
    #names = ["B_270k","C_270k","D_270k","CD_270k"] 
    names = ["A_34k","B_34k","C_34k","D_34k","CD_34k"]   
    training_mious = []
    for path in paths:
        event_acc = EventAccumulator(path, tf_size_guidance)
        event_acc.Reload()
        training_mious.append(event_acc.Scalars('epoch_mean_io_u'))

    # Show all tags in the log file
    #print(event_acc.Tags())

    steps = 8
    x = np.arange(1,steps+1,1)
    y = np.zeros([steps, 2])

    for j in range(len(training_mious)):
        for i in range(steps):
            y[i, 0] = training_mious[j][i][2] # value
        plt.plot(x, y[:,0], label=names[j])
    
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.xlim(1,8)
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.title("Validation after epochs")
    plt.legend(loc='lower right', frameon=True)
    plt.show()


if __name__ == '__main__':
    #log_file = ["C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/B270_autoencoder_LR1.0_19-Mar-2021-06-54-03/train/events.out.tfevents.1616133429.DESKTOP-7OUSENA.2188.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/C270_autoencoder_LR1.0_18-Mar-2021-08-42-13/train/events.out.tfevents.1616053405.DESKTOP-7GTLFRP.11300.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/D270_autoencoder_LR1.0_22-Mar-2021-19-34-04/train/events.out.tfevents.1616438116.DESKTOP-7GTLFRP.784.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/CD270_autoencoder_LR1.0_24-Mar-2021-07-25-12/train/events.out.tfevents.1616567183.DESKTOP-7GTLFRP.3444.3236.v2"
    #            ]#train270
    # log_file = ["C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/B270_autoencoder_LR1.0_19-Mar-2021-06-54-03/validation/events.out.tfevents.1616138531.DESKTOP-7OUSENA.2188.36826.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/C270_autoencoder_LR1.0_18-Mar-2021-08-42-13/validation/events.out.tfevents.1616058010.DESKTOP-7GTLFRP.11300.36826.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/D270_autoencoder_LR1.0_22-Mar-2021-19-34-04/validation/events.out.tfevents.1616442773.DESKTOP-7GTLFRP.784.36826.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/CD270_autoencoder_LR1.0_24-Mar-2021-07-25-12/validation/events.out.tfevents.1616570921.DESKTOP-7GTLFRP.3444.36826.v2"
    #            ]#valid270
    # log_file = ["C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/A34_autoencoder_LR1.0_10-Mar-2021-07-08-48/train/events.out.tfevents.1615356551.DESKTOP-7OUSENA.6268.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/B34_autoencoder_LR1.0_09-Mar-2021-20-13-23/train/events.out.tfevents.1615317228.DESKTOP-7OUSENA.6292.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/C34_autoencoder_LR1.0_10-Mar-2021-09-30-07/train/events.out.tfevents.1615365037.DESKTOP-7OUSENA.13468.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/D34_autoencoder_LR1.0_14-Mar-2021-19-40-11/train/events.out.tfevents.1615747274.DESKTOP-7OUSENA.7232.3236.v2",
    #             "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/CD34_autoencoder_LR1.0_18-Mar-2021-09-53-35/train/events.out.tfevents.1616057643.DESKTOP-7OUSENA.19768.3236.v2"
    #             ]
    log_file = ["C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/A34_autoencoder_LR1.0_10-Mar-2021-07-08-48/validation/events.out.tfevents.1615357159.DESKTOP-7OUSENA.6268.13230.v2",
                "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/B34_autoencoder_LR1.0_09-Mar-2021-20-13-23/validation/events.out.tfevents.1615318192.DESKTOP-7OUSENA.6292.13230.v2",
                "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/C34_autoencoder_LR1.0_10-Mar-2021-09-30-07/validation/events.out.tfevents.1615365903.DESKTOP-7OUSENA.13468.13230.v2",
                "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/D34_autoencoder_LR1.0_14-Mar-2021-19-40-11/validation/events.out.tfevents.1615748533.DESKTOP-7OUSENA.7232.13230.v2",
                "C:/Users/C201_ALES_NTB/source/repos/HGR_CNN/HGR_CNN/logs/CD34_autoencoder_LR1.0_18-Mar-2021-09-53-35/validation/events.out.tfevents.1616058409.DESKTOP-7OUSENA.19768.13230.v2"
                ]
    
    
    plot_tensorflow_log(log_file)
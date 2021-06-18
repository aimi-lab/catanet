import numpy as np
import os
import pandas as pd
import argparse
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
TEMPERATURE = 1.548991 # optimized temperature for calibration of Catnet

parser = argparse.ArgumentParser()

parser.add_argument(
    '--input',
    type=str,
    help='path to folder containing test results in csv'
)
args = parser.parse_args()
plt.rcParams['font.size'] = 16
cmap = plt.cm.get_cmap('tab20')  # 11 discrete colors

csv_files = glob.glob(os.path.join(args.input, '*.csv'))

# ensemble predictions for rsd and experience over folds
all_df = []
for file in csv_files:
    vname = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    df['gt_rsd'] = np.max(df['elapsed']) - df['elapsed']
    df['video'] = vname
    all_df.append(df)
    fig = plt.figure(figsize=[10, 5])
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1, 1], hspace=0.0, width_ratios=[4,1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:, 1])
    ax5 = fig.add_subplot(gs[:, 2])

    # legend for steps
    ax4.imshow(np.arange(11)[:, None], extent=[0, 1, 0, 11], cmap=cmap, interpolation='none', vmin=0, vmax=11)
    ax4.set_xticks([])
    ax4.yaxis.tick_right()
    phases = ['None','Inc', 'VisAgInj', 'Rhexis', 'Hydro', 'Phaco', 'IrAsp', 'CapsPol', 'LensImpl', 'VisAgRem', 'TonAnti']
    phases.reverse()
    ax4.set_yticks(ticks=np.arange(11)+0.5)
    ax4.set_yticklabels(phases)


    # create a second axes for the colorbar
    ax5.imshow(np.linspace(0,7,100)[:, None], extent=[0, 1, 0, 9], cmap='cividis', interpolation='none', vmin=0, vmax=7)
    ax5.set_xticks([])
    ax5.yaxis.tick_right()
    ax5.set_yticks([0,9])
    ax5.set_yticklabels(['senior', 'assistant'])

    #plt.subplots(3, 1, gridspec_kw={'height_ratios': [2.2, 1, 1]})
    ax1.plot(df['elapsed'], df['gt_rsd'], linewidth=2)
    ax1.plot(df['elapsed'], df['predicted_rsd'], linewidth=2)
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlim(left=np.min(df['elapsed']), right=np.max(df['elapsed']))
    ax1.legend(['ground-truth', 'predicted'])

    ax1.set_xticks([])
    # perform temperature scaling of experience predictions
    experience_cal = softmax(np.column_stack([df['predicted_senior'], df['predicted_assistant']])/TEMPERATURE, axis=-1)[:, 0]
    height = np.max(df['elapsed'])/15
    ax2.imshow(df['predicted_step'][None, :].astype(int), cmap=cmap,
                     extent=[0.0, np.max(df['elapsed']), -height, height], interpolation='none', vmin=0, vmax=11)
    ax3.imshow(experience_cal[None, :],cmap='cividis',
                     extent=[0.0, np.max(df['elapsed']), -height, height], interpolation='none', vmin=0, vmax=1)
    ax2.set_xticks([])
    ax2.set_ylabel('phase\n')
    ax3.set_ylabel('exp.\n')
    ax1.set_ylabel('RSD (min)')
    ax3.set_yticks([])
    ax2.set_yticks([])
    ax3.set_xlabel('elapsed time (min)')
    plt.savefig(os.path.join(args.input, vname + '.png'))
    plt.close()

all_df = pd.concat(all_df)
all_df['difference'] = all_df['gt_rsd'] - all_df['predicted_rsd']
all_df['absolute_error'] = np.abs(all_df['difference'])


def rsd_error(data_frame):
    rsd_2 = np.mean(data_frame[data_frame.gt_rsd < 2.0]['absolute_error'])
    rsd_5 = np.mean(data_frame[data_frame.gt_rsd < 5.0]['absolute_error'])
    rsd_all = np.mean(data_frame['absolute_error'])
    duration = np.max(data_frame['elapsed'])
    return pd.DataFrame([[rsd_2, rsd_5, rsd_all, duration]], columns=['rsd_2', 'rsd_5', 'rsd_all', 'duration'])


rsd_err = all_df.groupby(['video']).apply(rsd_error)
rsd_err_s = pd.concat([rsd_err.mean(), rsd_err.std()], axis=1)
rsd_err_s.columns = ['mean', 'std']
print('\nMacro Average RSD')
print(rsd_err_s)

plt.boxplot(rsd_err['rsd_all'])
plt.title('RSD Error')
plt.xticks([])
plt.ylabel('RSD MAE (min)')
plt.show()
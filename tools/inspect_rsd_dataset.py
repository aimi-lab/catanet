import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sd
import argparse


def minsec2min(minsec):
    minutes, seconds = minsec.split(':')
    return float(minutes) + float(seconds)/60


def main_others(input_path):
    assert os.path.isfile(os.path.join(input_path, 'start_end_labels.csv')), 'start end label file not found'
    labels = pd.read_csv(os.path.join(input_path, 'start_end_labels.csv'))
    durations = (labels['End'].apply(minsec2min) - labels['Start'].apply(minsec2min)).to_numpy()
    plot_duration_stats(durations)


def main_cataracts_train(input_path):
    labels = sorted(glob.glob(os.path.join(input_path, '*.csv')))
    assert len(labels) > 0, 'no label files found in ' + input_path
    durations = []
    for lab in labels:
        label = pd.read_csv(lab)['Steps'].to_numpy()
        label = np.concatenate((label, [0]))  # make sure the last value is 0 to find the edge
        start_frame = np.min(np.where((np.diff(label, prepend=0) != 0) & (label == 3))[0])
        end_frame = np.max(np.where((np.diff(label, prepend=0) != 0) & (label == 0))[0])
        durations.append((end_frame - start_frame)/(29.18*60))
    plot_duration_stats(durations)


def main_cataract101(input_path):
	labels = sorted(glob.glob(os.path.join(input_path, '*.csv')))
	assert len(labels) > 0, 'no label files found in ' + input_path
	durations = []
	for lab in labels:
		label_df = pd.read_csv(lab)
		label = np.concatenate((label_df['valid'].to_numpy(), [0]))  # make sure the last value is 0 to find the edge
		start_frame = np.min(np.where((np.diff(label, prepend=0) != 0) & (label == 1))[0])
		end_frame = np.max(np.where((np.diff(label, prepend=0) != 0) & (label == 0))[0])
		durations.append((end_frame - start_frame)/(25*60))
	plot_duration_stats(durations)


def plot_duration_stats(durations):
	sd.set_style("whitegrid")
	ax = sd.boxplot(x=durations)
	ax.set(xlabel='surgery duration [min]')
	plt.yticks([])
	plt.title('Surgery Duration')
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='path to folder containing label files or startend file.'
    )
    parser.add_argument(
        '--label',
        type=str,
        choices=['cataract101', 'CATARACTs', 'startend'],
        default='startend',
        help='type of label, if startend provide a csv file with fields {PatientID,Start,End}. '
             'Start/End in format XX:XX.'
    )
    args = parser.parse_args()
    if args.label == 'cataract101':
        main_cataract101(args.input)
    elif args.label == 'CATARACTs':
        main_cataracts_train(args.input)
    else:
        main_others(args.input)
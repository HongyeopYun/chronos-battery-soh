"""Readable supplementary chart from frozen desktop scores; no re-inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/c2-battery/one-step-4-reviewed-v3'


def main():
    data = pd.read_csv(OUT/'lastvalue_comparison.csv')
    scores = data[data.phase.eq('all')].set_index(['dataset','context_length','method'])
    groups = ['nasa_aging','calce_cs2_1c','mit_batch1','mit_batch2','mit_batch3','xjtu_2c','xjtu_3c']
    names = ['NASA','CALCE CS2','MIT batch 1','MIT batch 2','MIT batch 3','XJTU 2C','XJTU 3C']
    styles = [(64,'#90a4ae'),(32,'#2467ad'),(16,'#d67c23'),(8,'#6f4d9a'),(4,'#168575')]
    fig, axes = plt.subplots(1,2,figsize=(14,8),sharex=True,sharey=True)
    for ax,method in zip(axes,['zero-shot','head-only']):
        for i,(length,color) in enumerate(styles):
            values = [100*scores.loc[(g,length,method),'mse_skill_lastvalue'] for g in groups]
            ax.barh(np.arange(len(groups))+(i-2)*.16,values,height=.14,color=color,label=f'{length} inputs')
        ax.axvline(0,color='#424242',lw=1)
        ax.set_yticks(np.arange(len(groups)),names)
        ax.set_xlabel('MSE reduction vs LastValue (%)',fontsize=11)
        ax.set_title(method,fontsize=15,pad=14)
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Current SOH prediction: matched 64 / 32 / 16 / 8 / 4 inputs',y=.98,fontsize=16)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.935),ncol=5,frameon=False)
    fig.text(.53,.035,'Positive: better than LastValue | Negative: worse | Equal battery-record weighting',ha='center',fontsize=10,color='#555')
    fig.subplots_adjust(left=.11,right=.98,bottom=.13,top=.82,wspace=.06)
    fig.savefig(OUT/'comparison_readable.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()

"""Plot the requested deterministic checkpoint-transfer sweep."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('processing/evaluations/motiontrans_vjepa21_deterministic_epochs_v1')
s=json.loads((root/'summary.json').read_text())
rows=[(int(epoch),r['validation']) for epoch,r in s['results'].items()]
rows.append((28,s['epoch28_reference']['validation']));rows.sort()
fig,ax=plt.subplots(figsize=(7.5,4.4),layout='constrained')
ax.errorbar([e for e,_ in rows],[m['mae'] for _,m in rows],
 yerr=[m['mae_head_seed_std'] for _,m in rows],fmt='o-',capsize=4,color='#126e82',
 label='V-JEPA deterministic 32D (mean ± head-seed SD)')
reference=s['dino_epoch7_reference']['validation']['mae']
ax.axhline(reference,color='#b54835',linestyle='--',label=f'DINO epoch-7 32D reference ({reference:.5f})')
ax.set(xlabel='V-JEPA source-training checkpoint epoch',ylabel='MotionTrans normalized MAE (lower is better)',
 title='MotionTrans transfer across deterministic V-JEPA checkpoints',xticks=[5,10,15,20,25,28])
ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=8.5,loc='best')
fig.text(.02,-.035,'Three fitted head seeds; frozen encoders; same 32 → 512 → 512 → 512 → 6 head.\nError bars describe head-fit variation; paired episode-bootstrap intervals are in analysis.md.',fontsize=8)
for extension in ['png','pdf']:
 fig.savefig(root/f'transfer_by_epoch.{extension}',dpi=180,bbox_inches='tight')
print(root/'transfer_by_epoch.png')

"""Common reproducible drawing functions for ResLuma Figures 4-19, excluding 2.

Qualitative panels are recomposed from the individual raster image tiles used in
paper's final PDF figures. Plot functions read the accompanying numeric CSV/JSON.
They do NOT execute any model inference or regenerate model outputs.
"""
from pathlib import Path
from collections import defaultdict
import json,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics

BASE=Path(__file__).resolve().parent.parent
DATA=BASE/'source_data'
TABLES=BASE/'tables'
OUT=BASE/'generated_figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
 'font.size':10,'axes.titlesize':10,'axes.labelsize':9.5,'xtick.labelsize':8.5,
 'ytick.labelsize':8.5,'legend.fontsize':8,'axes.linewidth':.75,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def readcsv(file,folder=DATA):
 with open(folder/file,encoding='utf-8-sig',newline='') as f:
  return list(csv.DictReader(f))

def save_plot(n,fig):
 fig.savefig(OUT/f'Figure_{n:02d}.pdf',bbox_inches='tight',pad_inches=.06)
 fig.savefig(OUT/f'Figure_{n:02d}.png',dpi=175,bbox_inches='tight',pad_inches=.06)
 plt.close(fig)

def style(ax):
 ax.grid(alpha=.28,ls=':',lw=.6)
 ax.set_axisbelow(True)

def compose_qualitative(n):
 """Arrange extracted original image tiles with fresh editable vector labels.

 No fabricated sample images. Relative image content and method order match the
 published figure, but whitespace/font positions can differ slightly.
 """
 cfg=json.loads((BASE/'figure_index.json').read_text(encoding='utf8'))['composite_configs'][str(n)]
 folder=BASE/'inputs'/'tiles'/f'figure_{n:02d}'
 rows=cfg['rows']
 ncols=cfg['ncols']
 nrows=len(rows)
 page_w=550.0
 margin_left=32 if cfg.get('show_left') else 9
 margin_right=8.0
 gap_x=4.0
 imgw=(page_w-margin_left-margin_right-(ncols-1)*gap_x)/ncols
 # Each row keeps its own original image aspect. Original panels may have mixed ratios.
 rowheights=[]
 for ridx,nn in enumerate(rows):
  path=folder/f'r{ridx+1:02d}_c01.png'
  with Image.open(path) as img:
   rowheights.append(imgw*img.height/img.width)
 # Vertical title/scene spacing below rows for qualitative comparisons.
 title_h=14
 scene_gap=19
 row_gap=5
 if cfg['title_mode']=='bottom':
  total_h=sum(rowheights)+nrows*title_h+(nrows-1)*row_gap+len(cfg['scenes'])*scene_gap+10
 else:
  total_h=sum(rowheights)+nrows*row_gap+(nrows-1)*scene_gap+29
 can=Canvas(str(OUT/f'Figure_{n:02d}.pdf'),pagesize=(page_w,total_h),pageCompression=1)
 y=total_h-9
 scene_nrows=cfg['scene_nrows']
 labels=cfg['labels']
 for ridx,nn in enumerate(rows):
  if ridx and ridx%scene_nrows==0: y-=scene_gap
  scene_index=ridx//scene_nrows
  if cfg['title_mode']=='top' and ridx==0:
   for col,lab in enumerate(labels):
    x=margin_left+col*(imgw+gap_x)+imgw/2
    can.setFont('Helvetica',8.3)
    can.drawCentredString(x,y-5,str(lab))
   y-=17
  image_h=rowheights[ridx]
  imgy=y-image_h
  # If only 5 images on second row of BIMEF, center them across 6-column grid.
  shift=(ncols-nn)*(imgw+gap_x)/2
  for j in range(nn):
   im=Image.open(folder/f'r{ridx+1:02d}_c{j+1:02d}.png').convert('RGB')
   x=margin_left+shift+j*(imgw+gap_x)
   can.drawImage(ImageReader(im),x,imgy,width=imgw,height=image_h,preserveAspectRatio=False)
  if cfg.get('show_left') and ridx%scene_nrows==0:
   can.saveState();can.setFont('Helvetica',8.1)
   can.translate(12,imgy+image_h*.5);can.rotate(90)
   can.drawCentredString(0,0,cfg['scenes'][scene_index]);can.restoreState()
  y=imgy
  if cfg['title_mode']=='bottom':
   y-=13
   row_labels=labels[ridx%scene_nrows]
   for j,lab in enumerate(row_labels):
    x=margin_left+shift+j*(imgw+gap_x)+imgw/2
    can.setFont('Helvetica-Bold' if lab.startswith('ResLuma') else 'Helvetica',7.8)
    can.drawCentredString(x,y+2,str(lab))
  else:
   # no repeated method labels for each scene, consistent with final paper figure.
   pass
  y-=row_gap
 can.showPage();can.save()
 # PNG rendering uses the generated vector PDF itself; no font files embedded in archive.
 import fitz
 doc=fitz.open(OUT/f'Figure_{n:02d}.pdf')
 pix=doc[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False)
 pix.save(OUT/f'Figure_{n:02d}.png')
 doc.close()

def fig6():
 vals=readcsv('Table_3_latency_FPS.csv',TABLES)
 methods=list(dict.fromkeys(r['method'] for r in vals))
 res=['512x512','720p','1080p','4K']
 fig,ax=plt.subplots(figsize=(7.5,3.9))
 for meth in methods:
  data=[next((r for r in vals if r['method']==meth and r['resolution']==r0),{}) for r0 in res]
  yy=[float(r['fps']) if r.get('fps') and r['fps']!='OOM' else np.nan for r in data]
  ax.plot(range(4),yy,marker='*' if meth.startswith('ResLuma') else 'o',
    markersize=8 if meth.startswith('ResLuma') else 3.5,
    linewidth=2.3 if meth.startswith('ResLuma') else 1,label=meth)
 ax.set_yscale('log');ax.set_xticks(range(4),['512²','720p','1080p','4K']);ax.set_ylabel('GPU FPS (log)');ax.set_xlabel('Input resolution')
 ax.grid(alpha=.26,axis='y');ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',frameon=False,ncol=1)
 fig.subplots_adjust(right=.79,left=.12,bottom=.17);save_plot(6,fig)

def fig8():
 dat=readcsv('Table_4_core_ablation.csv',TABLES)
 fig,axes=plt.subplots(1,2,figsize=(7.4,2.9))
 for ax,metric,title in zip(axes,['LOL_PSNR_dB','LOL_SSIM'],['(a) PSNR','(b) SSIM']):
  yy=np.array([float(r[f'{metric}_mean']) for r in dat]);sd=np.array([float(r[f'{metric}_std']) for r in dat])
  base=yy.min()-(.9 if 'PSNR' in metric else .05)
  bars=ax.bar(range(4),yy-base,bottom=base,yerr=sd,capsize=3,width=.65,edgecolor='#333333',linewidth=.5)
  ax.set_xticks(range(4),['A0','A1','A2','A3']);ax.set_title(title);ax.set_ylabel('PSNR (dB)' if 'PSNR' in metric else 'SSIM');style(ax)
  ax.set_ylim(yy.min()-(.9 if 'PSNR' in metric else .05),yy.max()+(0.7 if 'PSNR' in metric else .04))
  for i,v in enumerate(yy):ax.text(i,v+sd[i]+(.03 if 'PSNR' in metric else .002),f'{v:.3f}',ha='center',fontsize=8.3)
 fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.86,wspace=.30)
 fig.savefig(OUT/"Figure_08.pdf")
 fig.savefig(OUT/"Figure_08.png",dpi=175)
 plt.close(fig)

def fig11():
 dat=readcsv('Table_5_curve_and_eta.csv',TABLES)
 fig,axes=plt.subplots(2,2,figsize=(7.4,5.2))
 for i,grp in enumerate(['curve_order_k','correction_strength_eta']):
  rows=[r for r in dat if r['experiment']==grp]
  xx=[float(r['setting']) for r in rows]
  for j,(key,ylabel) in enumerate([('PSNR_dB','PSNR (dB)'),('SSIM','SSIM')]):
   ax=axes[i,j]
   y=[float(r[f'{key}_mean']) for r in rows]
   err=[float(r[f'{key}_std']) for r in rows]
   ax.errorbar(xx,y,yerr=err,marker='o',capsize=3,markersize=4,lw=1.4)
   ax.set_title(f'({chr(ord("a")+2*i+j)}) {ylabel} vs. '+('k' if i==0 else 'eta'))
   ax.set_xlabel('Curve order k' if i==0 else 'Correction strength eta');ax.set_ylabel(ylabel);style(ax)
   if i==0:ax.set_xscale('log',base=2);ax.set_xticks(xx,[str(int(v)) for v in xx])
   else:ax.set_xticks(xx)
 fig.tight_layout(h_pad=1.4);save_plot(11,fig)

def fig13():
 dat=readcsv('beta_stats.csv')
 fig,axes=plt.subplots(1,3,figsize=(7.4,2.8),sharey=True)
 for j,(key,title) in enumerate([('B','Mean luminance'),('H','Highlight ratio'),('D','Dark ratio')]):
  ax=axes[j]
  for dataset,mark in [('LOL/test','o'),('BIMEF','^')]:
   r=[x for x in dat if x['dataset']==dataset]
   ax.scatter([float(x[key]) for x in r],[float(x['beta_adaptive']) for x in r],s=10,marker=mark,alpha=.72,label=dataset)
  ax.set_title(f'({chr(ord("a")+j)}) {title}');ax.set_xlabel(title);ax.set_ylim(.33,1.03);style(ax)
 axes[0].set_ylabel('Adaptive beta');axes[0].legend(fontsize=7)
 fig.tight_layout();save_plot(13,fig)

def fig16():
 a=json.loads((DATA/'figure16_LOL_sweep.json').read_text())['grid']
 b=json.loads((DATA/'figure16_BIMEF_sweep.json').read_text())['grid']
 ts=sorted(set(float(x['t']) for x in a));gs=sorted(set(float(x['gamma']) for x in a))
 def mat(rows,key,sgn=1):
  m=np.zeros((len(gs),len(ts)))
  for r in rows:m[gs.index(float(r['gamma'])),ts.index(float(r['t']))]=sgn*float(r[key])
  return m
 panels=[('LOL: ΔPSNR (dB)',mat(a,'d_psnr')),('LOL: ΔSSIM',mat(a,'d_ssim')),
         ('BIMEF: ΔNIQE',mat(b,'d_niqe',-1)),('BIMEF: ΔCLIP-IQA',mat(b,'d_clipiqa'))]
 fig,axes=plt.subplots(2,2,figsize=(7.5,5.3))
 for idx,(ax,(title,m)) in enumerate(zip(axes.ravel(),panels)):
  lim=max(abs(m.max()),abs(m.min()),1e-6)
  im=ax.imshow(m,cmap='RdBu_r',aspect='auto',vmin=-lim,vmax=lim)
  ax.set_title(f'({chr(ord("a")+idx)}) {title}')
  ax.set_xticks(range(len(ts)),[f'{x:.2f}' for x in ts]);ax.set_yticks(range(len(gs)),[f'{x:.3f}' for x in gs]);ax.set_xlabel('Threshold t');ax.set_ylabel('Strength gamma')
  for j in range(len(gs)):
   for k in range(len(ts)):ax.text(k,j,f'{m[j,k]:+.3f}',fontsize=5.8,ha='center',va='center')
  ax.add_patch(Rectangle((ts.index(.4)-.5,gs.index(.075)-.5),1,1,fill=False,ec='black',lw=1.35))
  fig.colorbar(im,ax=ax,fraction=.043,pad=.035).ax.tick_params(labelsize=7)
 fig.subplots_adjust(left=.09,right=.95,bottom=.095,top=.94,hspace=.55,wspace=.36);save_plot(16,fig)

def fig18():
 dat=readcsv('Table_9_kernel_size.csv',TABLES)
 fig,axes=plt.subplots(1,2,figsize=(7.4,3))
 for i,ax in enumerate(axes):
  for r in dat:
   x=float(r['parameters'] if i==0 else r['GPU_latency_ms_mean_or_value'])
   y=float(r['LOL_PSNR_dB_mean_or_value']);sd=float(r['LOL_PSNR_dB_std_if_reported'])
   kw={'marker':'*','markersize':10} if r['kernel']=='3x3' else {'marker':'o','markersize':5}
   ax.errorbar(x,y,yerr=sd,fmt='',capsize=3,lw=.85,**kw)
   ax.annotate(r['kernel'],(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
  ax.set_title(['(a) Quality vs. parameters','(b) Quality vs. latency'][i])
  ax.set_xlabel(['Trainable parameters','GPU latency @ 512x512 (ms)'][i]);ax.set_ylabel('LOL PSNR (dB)');style(ax)
 fig.tight_layout();save_plot(18,fig)

COMPOSITES={4,5,7,9,10,12,14,15,17,19}
NUMERIC={6:fig6,8:fig8,11:fig11,13:fig13,16:fig16,18:fig18}
def generate(n):
 if n in COMPOSITES:compose_qualitative(n)
 elif n in NUMERIC:NUMERIC[n]()
 else:raise ValueError(f'Figure {n} is handled by original standalone scripts 01, 03; Figure 02 excluded.')
 print('Generated Figure',n)

"""Standalone scientific output figures; no ObsPy dependency."""
import json
from pathlib import Path
import numpy as np


def plot_result(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from .mechanism import moment_matrix
    output=Path(output);report=json.loads((output/'result.json').read_text())
    moment=report.get('moment_tensor_Nm',report.get('moment_tensor_normalized'))
    matrix=moment_matrix(torch.tensor(moment,dtype=torch.float64)).numpy()
    matrix/=np.linalg.norm(matrix)
    z=np.linspace(-1,1,401);x,y=np.meshgrid(z,z);r=np.hypot(x,y)
    angle=2*np.arcsin(np.minimum(r,1)/np.sqrt(2));az=np.arctan2(x,y)
    rays=np.stack([np.sin(angle)*np.cos(az),np.sin(angle)*np.sin(az),np.cos(angle)],axis=-1)
    amplitude=np.einsum('...i,ij,...j->...',rays,matrix,rays)
    amplitude[r>1]=np.nan
    fig,ax=plt.subplots(figsize=(7,7))
    ax.contourf(x,y,amplitude,levels=[-2,0,2],colors=['#c9ddf0','#efbe8a'])
    ax.contour(x,y,amplitude,levels=[0],colors='#393939',linewidths=1)
    ax.add_patch(plt.Circle((0,0),1,fill=False,color='#333333',lw=1.5))
    for label,pos in [('N',(0,1.08)),('E',(1.08,0)),('S',(0,-1.08)),('W',(-1.08,0))]:
        ax.text(*pos,label,ha='center',va='center')
    if 'azimuth_deg' in report:
        azimuth=np.deg2rad(report['azimuth_deg']);takeoff=np.deg2rad(np.array([np.nan if v is None else v for v in report['takeoff_angle_deg']]))
        up=takeoff>np.pi/2;takeoff=np.where(up,np.pi-takeoff,takeoff);azimuth=azimuth+up*np.pi
        radius=np.sqrt(2)*np.sin(takeoff/2)
        px,py=radius*np.sin(azimuth),radius*np.cos(azimuth)
        probabilities=np.asarray(report['probability_u'])
        valid=np.asarray(report['takeoff_gradient_valid'])&np.asarray(report['polarity_input_valid'])
        for i in np.flatnonzero(valid):
            ax.scatter(px[i],py[i],s=55,c='#d66c21' if probabilities[i]>.5 else '#246da3',edgecolor='white',lw=.7,zorder=4)
            ax.annotate(str(i+1),(px[i],py[i]),xytext=(3,3),textcoords='offset points',fontsize=7)
    plane=report['nodal_planes'][0]
    ax.set_title(f"{report['event_id']}\nstrike {plane['strike']:.1f}°  dip {plane['dip']:.1f}°  rake {plane['rake']:.1f}°\ndepth {report['depth_km']:.1f} km | {report['mode']}",pad=20)
    ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),aspect='equal');ax.axis('off')
    fig.text(.5,.025,'Lower hemisphere, equal area. Orange: U; blue: D.\nStation numbers follow station_ids in result.json.',ha='center',fontsize=9)
    fig.savefig(output/'mechanism.png',dpi=180,bbox_inches='tight');fig.savefig(output/'mechanism.pdf',bbox_inches='tight');plt.close(fig)
    file=output/'waveform_fit.npz'
    if file.exists():
        data=np.load(file);observed=data['observed'];predicted=data['predicted'];times=data['relative_times_s'];valid=data['valid']
        n=len(observed);fig,axes=plt.subplots(n,5,figsize=(15,max(4,1.6*n)),squeeze=False)
        labels=report['segment_order']
        for i in range(n):
            # One scale per station for display only, preserving component ratios.
            scale=max(np.abs(observed[i]).max(),np.abs(predicted[i]).max(),1e-30)
            for j in range(5):
                ax=axes[i,j]
                if valid[i,j]:
                    ax.plot(times[i,j],observed[i,j]/scale,color='#292929',lw=.9,label='Observed')
                    ax.plot(times[i,j],predicted[i,j]/scale,color='#cd702c',lw=.9,label='Synthetic')
                else:ax.text(.5,.5,'Outside record',ha='center',transform=ax.transAxes,fontsize=8)
                ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7)
                if i==0:ax.set_title(labels[j],fontsize=10)
                if j==0:ax.set_ylabel(report['station_ids'][i],fontsize=8)
                if i==n-1:ax.set_xlabel('Seconds since origin',fontsize=8)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f"CAP waveform fit | VR {100*report['variance_reduction']:.2f}% | display scale shared by station",y=1.005)
        fig.tight_layout();fig.savefig(output/'waveform_fit.png',dpi=160,bbox_inches='tight');fig.savefig(output/'waveform_fit.pdf',bbox_inches='tight');plt.close(fig)

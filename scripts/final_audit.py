"""Build inspectable phase metrics, archive identities and run summaries."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import grtm
from scripts.train_body import load,evaluate
from seismic_joint.networks import body_network
from seismic_joint.schema import BODY_PHASES,EARTH_MODELS
from seismic_joint.mechanism import tensor_angle_deg


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda:0';state=torch.load(ROOT/'assets/body_travel_expert.pt',map_location=device,weights_only=True)
    model=body_network(state['config']).to(device);model.load_state_dict(state['state_dict'])
    test=load('test',device,state['metadata']);te,ae,ix=evaluate(model,test,all_rows=True)
    rows=[]
    for mi,name in enumerate(EARTH_MODELS):
        for pi,phase in enumerate(BODY_PHASES):
            group=(test['earth'][ix]==mi)&(test['phase'][ix]==pi)
            for scope in ['full_domain','regional_cap']:
                use=group
                if scope=='regional_cap':use=use&(test['x'][ix,0]<=60)&(test['x'][ix,1]<=500)
                row=dict(model=name,phase=phase,scope=scope,n=int(use.sum()))
                if use.any():row.update(time_mae_s=float(te[use].mean()),time_p95_s=float(torch.quantile(te[use],.95)),
                    time_max_s=float(te[use].amax()),angle_mae_deg=float(ae[use].mean()),
                    angle_p95_deg=float(torch.quantile(ae[use],.95)),angle_max_deg=float(ae[use].amax()))
                rows.append(row)
    fields=['model','phase','scope','n','time_mae_s','time_p95_s','time_max_s','angle_mae_deg','angle_p95_deg','angle_max_deg']
    with (ROOT/'results/body_phase_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    synthetic=json.loads((ROOT/'results/synthetic_joint/result.json').read_text())
    truth=json.loads((ROOT/'examples/synthetic/truth.json').read_text())
    real=json.loads((ROOT/'results/csndata_polarity/result.json').read_text())
    checks=json.loads((ROOT/'results/pipeline_validation.json').read_text())
    assert checks['status']=='pass'
    assert synthetic['depth_km']==truth['depth_km']
    angle=tensor_angle_deg(synthetic['moment_tensor_Nm'],truth['moment_tensor_Nm'])
    assert angle<.1 and synthetic['variance_reduction']>.99
    identity={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for folder in ['assets','seismic_joint','scripts','tests'] for p in sorted((ROOT/folder).rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.pt','.jit','.onnx','.py','.json')}
    manifest=dict(status='complete',grtm_version=grtm.__version__,torch_version=torch.__version__,files=identity,
        private_only=True,runtime_taup=False,body_dnn='39 independent Earth/phase experts with time/gradient supervision',
        input_gradient_angle_convention='atan2(R/(R-h)*abs(dT/dd),-dT/dh);0down,90horizontal,180up',
        green_geometry_scope='fixed geometry; physical depth grid; optional approximate interpolated bank',
        synthetic_recovery=dict(depth_km=synthetic['depth_km'],moment_magnitude=synthetic['moment_magnitude'],
            tensor_angle_deg=angle,variance_reduction=synthetic['variance_reduction']),
        real_example=dict(event_id=real['event_id'],stations=real['stations_used'],
            mismatches=real['polarity_mismatches'],cnn_manual=real.get('cnn_manual_audit'),known_mechanism_truth=False),
        test_domain_depth_km=[0,200],test_domain_distance_km=[0,3000],recommended_cap_distance_km=[10,500])
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    summary={k:v for k,v in manifest.items() if k!='files'}
    summary['holdout_records_by_phase']=rows
    (ROOT/'results/final_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['# 完成情况与核对结果','','正常推理使用 DNN 走时，不导入 ObsPy/TauP。',
        '新增体波专家和面波模型训练完成；原有首至 P 和初动 CNN 权重保持不变。','',
        '## 新增体波网络独立测试','','下表是全域平均走时误差（秒），不是最大误差。',
        '完整平均、P95、最大走时/角度误差及区域子集见 `results/body_phase_accuracy.csv`。','',
        '| 震相 | iasp91 | ak135 | PREM |','|---|---:|---:|---:|']
    for phase in BODY_PHASES:
        values=[next(r for r in rows if r['model']==earth and r['phase']==phase and r['scope']=='full_domain') for earth in EARTH_MODELS]
        lines.append('| '+phase+' | '+' | '.join(f"{v['time_mae_s']:.4f}" for v in values)+' |')
    lines.extend(['','上行 p/s 和部分多路径/反射震相仍有较大尾部误差。有效性标记不等于误差证书；不能据此宣称任意震相的全域高精度 TauP 替代。',
        'CAP 区域窗口主要使用保留的首至 P、新首至 S 和 Rayleigh/Love 群速度。','',
        '## 合成与真实数据流程检验','',
        f"GRTM 合成真值：走向 35°、倾角 50°、滑动角 −70°、Mw 4、深度 10 km。恢复深度 {synthetic['depth_km']:.1f} km，Mw {synthetic['moment_magnitude']:.6f}，归一化矩张量夹角 {angle:.6f}°，VR {100*synthetic['variance_reduction']:.5f}%。这是匹配模型的合成检验，不是实际地震反演准确率。",'',
        f"真实示例 {real['event_id']}：{real['stations_used']} 个台站，CNN/人工初动核对 {real.get('cnn_manual_audit',{}).get('agreement')}/{real.get('cnn_manual_audit',{}).get('n')}；初动机制极性不符 {real['polarity_mismatches']} 个。没有权威机制真值。",'',
        '## 梯度与积分核对','',
        '`results/pipeline_validation.json` 验证禁用 ObsPy/TauP 后的网络、出射角梯度链路，以及原生 GRTM 与线性 Green 库的源导数一致性。',
        '`results/joint_gradient_validation.json` 对完整联合损失的走向、倾角、滑动角、幅值及三组时移进行差分核对。',
        '`results/green_quadrature.json` 单独加密波数间隔和扩展高波数截断；本例滤波后相对 L2 差异均低于 2%，仅适用于记录的区域案例。',
        '原生 GRTM 源深几何不作精确自动微分承诺；默认使用物理深度网格。', '',
        '运行和数据格式说明见 `README.md`。模型/代码 SHA256 见 `manifest.json`。'])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

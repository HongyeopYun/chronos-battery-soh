"""Measure real XPU Full/LoRA backward passes using source data only."""
import sys,time,json,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from chronos import Chronos2Pipeline
from src.c2_v4_experiments import DirectRUL
from src.c2_rul_context_nominal import load,assemble

if __name__=='__main__':
    torch.set_num_threads(4)
    assert torch.xpu.is_available()
    cells,_,_=load(ROOT)
    a=assemble(cells,'mit_batch1',['b1c0','b1c1'],'16',length=128,start=128)
    pipe=Chronos2Pipeline.from_pretrained(str(ROOT/'models/huggingface/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c'),device_map='xpu',dtype=torch.float32,local_files_only=True)
    rows=[]
    for mode in ['full','lora']:
        model=DirectRUL(pipe.model,mode,float(np.median(a['target'])),float(np.mean(a['target'])))
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=1e-4)
        x=torch.as_tensor(a['context'][:16],device='xpu');y=torch.as_tensor(a['target'][:16],device='xpu',dtype=torch.float32)
        times=[]
        for i in range(13):
            torch.xpu.synchronize();start=time.monotonic()
            opt.zero_grad(set_to_none=True)
            pred=model(x);loss=torch.nn.functional.huber_loss(pred/model.scale,y/model.scale)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            torch.xpu.synchronize();times.append(time.monotonic()-start)
            if i==5:
                assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad)
            print(mode,i,round(times[-1],3),float(loss),flush=True)
        rows.append(dict(mode=mode,seconds_per_step=float(np.mean(times[3:])),trainable=sum(p.numel() for p in params),peak_bytes=torch.xpu.max_memory_allocated()))
        del model,opt,params,x,y;gc.collect();torch.xpu.empty_cache()
    out=ROOT/'results/c2-battery/rul-context-nominal80-adaptation-xpu-v1/.runtime';out.mkdir(parents=True,exist_ok=True)
    (out/'benchmark.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows),flush=True)

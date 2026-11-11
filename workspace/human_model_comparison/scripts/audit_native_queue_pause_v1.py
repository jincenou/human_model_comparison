"""Record the user's conditional pause with observed persistent-job evidence."""
from human_asset_benchmark.paths import expand as _hmc_expand
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

H=Path(__file__).resolve().parents[1];P=H/'model_preparation/20261110';Q=H/'runs/human_native_queue_v1'
def read(p):return json.loads(Path(p).read_text())
def live(pid):
    try:return not Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].startswith('Z ')
    except OSError:return False
def write(p,d):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
    q=read(Q/'state.json');assert live(q['pid'])
    completed=[j for j in q['jobs'] if j['status']=='native_passed']
    active=[j for j in q['jobs'] if j['status']=='running']
    errors=[j for j in q['jobs'] if j['status']=='failed']
    assert len({j['method']for j in completed})>=2,'Need demonstrated completion from multiple methods'
    assert any(j['sample_id']!='HN0004' for j in completed),'Need demonstrated successor completion'
    assert active and all(live(j['child_pid'])for j in active),'Need live persistent workers'
    assert not any(s['status']=='controller_slot_error'for s in q['slots'].values())
    weights=read(P/'state/modelscope_weights.json')
    installers={name:read(P/'state'/name)for name in ['exavatar_system_environment.json','jacobian_train_system_environment.json']}
    observed=[]
    for name,data in [('modelscope',weights),*installers.items()]:
        pid=data.get('pid');observed.append(dict(name=name,pid=pid,alive=live(pid),status=data.get('status')))
        assert live(pid) or data.get('status') not in ('running','installing'),name
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    tmux=subprocess.check_output(['tmux','-S',str(P/'state/preparation.sock'),'list-panes','-a','-F','#S #{pane_pid} #{pane_current_command}'],text=True)
    report=dict(status='user_pause_condition_met',observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        user_clarification='现有可运行方法稳定在后台执行后暂停，下载和安装继续',
        controller_pid=q['pid'],queue_state=str(Q/'state.json'),counts=q['counts'],active=active,
        completed=[{k:j[k]for k in ['key','method','sample_id','gpu','exit_code','worker_record']}for j in completed],failures=errors,
        gpu_pool=q['gpu_pool'],slots=q['slots'],gpu_inventory=gpu,gpu_processes=processes,tmux_sessions=tmux,
        preparation_processes=observed,downloads=[{k:j.get(k)for k in ['filename','status','bytes_downloaded','bytes']}for j in weights['jobs']],
        free_gib={str(p):shutil.disk_usage(p).free/1024**3 for p in [H,Path(_hmc_expand('${HMC_SYSTEM_ROOT}'))]},
        semantic_note='Human queue performs native generation, extraction and reload, not optimizer training. Existing rigid jobs continue. ExAvatar/Jacobian are not yet training.',
        protections=['shared GPU leases held through descendants','legacy unshared 6/7 controller reservation','20GiB launch floor and 11GiB stage floor','two failures stop further jobs of that method','no implicit retry or overwrite','new static method pilot gates'],
        pending=['Reference qualification and mask review','SplatAD physical-time audit','Real-camera registration for generated assets and Ours',
          'Metric implementation/numerical tests/config freeze','ExAvatar/Jacobian remaining dependencies and real-human pilots',
          'FlexiAvatar, Hunyuan, Recon and bounded StreetGS human pilots','Fixed-frontend-cache backend ablation','Paired metrics, coverage and final gallery'],
        scores_produced=False,background_processes_left_running=True)
    write(H/'state/pause_handoff_v1.json',report)
    progress=read(H/'state/human_experiment_progress_v2.json')
    progress.update(updated_at_utc=report['observed_at_utc'],status='pause_requested_background_running',
        latest_user_steering=report['user_clarification'],pause_handoff=str(H/'state/pause_handoff_v1.json'))
    progress['pause_instruction'].update(condition_met=True,user_clarification=report['user_clarification'],
        condition='Stable existing native generation/extraction/reload queue, with verified completed jobs and automatically launched successors; preparation continues.',evidence=str(H/'state/pause_handoff_v1.json'))
    progress['ordered_stages'][1]['counts']['runtime_render_passed']=1
    progress['ordered_stages'][3].update(status='native_production_running_under_pilot_gates',remaining=report['pending'][4:6])
    progress['ordered_stages'][3]['evidence']+=[str(Q/'state.json')]
    progress['ordered_stages'][4].update(status='OmniRe_physical_time_pilot_passed_common_qualification_pending',
        remaining='All100 OmniRe sample/camera clock gaps audited; physical query endpoint equality and projection passed on HN0004. SplatAD timing, reference qualification and real-world registration remain pending.')
    progress['ordered_stages'][4]['evidence']+=['reports/model_time_audit_v1/omnire.json','exports/common_camera_human_time_v2/omnire/HN0004/validation.json']
    progress['ordered_stages'][6].update(status='native_candidate_production_in_background_formal_metrics_pending',evidence=[str(Q/'state.json')])
    progress['parallel_preparation'].update(background_panes=tmux,download_current=report['downloads'],
        disk_free_gib=report['free_gib'][str(H)],system_disk_free_gib_last=report['free_gib'][_hmc_expand('${HMC_SYSTEM_ROOT}')],
        native_queue=str(Q/'state.json'),omnire_native_pilot=read(H/'state/omnire_human_time_v2.json'))
    progress['next_action']='On resume: inspect queue/installer/download results and failures first; preserve existing outputs and continue pending qualification/method/metric/ablation tasks.'
    write(H/'state/human_experiment_progress_v2.json',progress)
    print(json.dumps(dict(report=str(H/'state/pause_handoff_v1.json'),counts=q['counts'],active=[j['key']for j in active],preparation=observed)))
if __name__=='__main__':main()

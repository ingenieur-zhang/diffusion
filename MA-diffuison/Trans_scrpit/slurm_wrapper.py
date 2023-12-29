import wandb
import subprocess
import os
import json
import time
import myconfig1
from datetime import datetime


# check echo from 'sacct' to tell the job status
def check_status(status):
    rtn = 'RUNNING'
    
    lines = status.split('\n')
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        if 'FAILED' in line:
            rtn = 'FAILED'
            break
        elif 'COMPLETED' not in line:
            rtn = 'PENDING'
            break
    else:
        rtn = 'COMPLETED'
        
    return rtn


def wrap_task(config=None):
    # recieve config for this run from Sweep Controller
    with wandb.init(config=config):
        agent_id = wandb.run.id
        agent_dir = wandb.run.dir
        config = dict(wandb.config)
        
        # wait until available pipe slot
        while True:
            cmd = f"squeue -n {myconfig1.project_name}"
            status = subprocess.check_output(cmd, shell=True).decode()
            lines = status.split('\n')[1:-1]
            if len(lines) <= myconfig1.pool_size:
                break
            else:
                time.sleep(60)  # sdil
        
        # then build up the slurm script
        job_script = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig1.project_name}
#SBATCH --partition=dev_gpu_4
#SBATCH --gres=gpu:1
#SBATCH --error={myconfig1.log_path}%x.%j.err
#SBATCH --output={myconfig1.log_path}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig1.e_mail}
#SBATCH --export=ALL
#SBATCH --time=00:30:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig1.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig1.conda_env}/lib

job_id=$SLURM_JOB_ID
python {myconfig1.train_script_name} $job_id '{json.dumps(config)}' {agent_id} {agent_dir}
"""
        
        # Write job submission script to a file
        with open(myconfig1.slurm_scripts_path + f"{wandb.run.id}.sbatch", "w") as f:
            f.write(job_script)
        
        # Submit job to Slurm system and get job ID
        cmd = "sbatch " + myconfig1.slurm_scripts_path + f"{wandb.run.id}.sbatch"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        job_id = output.split()[-1]
        wandb.log({
            "job_id" : job_id,
        })
        return job_id
        
           
if __name__ == '__main__':
    rtn = wrap_task()
    print(f'******************************************************* Process Finished with code {rtn}')
    wandb.finish()

1. Enter data_set
python ../scripts/expand_eight_directions.py center_gro around_gro charge

2. Enter data_set/refer_QM
find ./ -name "sbatch.sh" -exec dirname {} \; > task_dirs.list
sbatch calc_qm.sh
```
#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=
#SBATCH --account=
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1

MAX_JOBS=100
DELAY=30s
JOB_TAG="zd-qm"

while read dir; do
    while true; do
        running_jobs=$(squeue -u $USER -h -t RUNNING,PENDING -o "%j" | grep "${JOB_TAG}" | wc -l)
        
        if (( running_jobs < MAX_JOBS )); then
            echo "Submit the task: $dir"
            (cd "$dir" && sbatch sbatch.sh *.com >/dev/null)
            break
        else
            echo "Currently running tasks: $running_jobs, waiting..."
            sleep $DELAY
        fi
    done
done < task_dirs.list

echo "All tasks have been submitted"
```

3. Enter data_set/refer_QM -> AA2CG
sbatch ../../script/mapping.sh task_dirs.list failed_task_dirs.list
```
#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=
#SBATCH --account=
#SBATCH -N 1
#SBATCH -n 124
#SBATCH -c 1

MAX_JOBS=124
FAILED_FILE=$2

cat $1 | xargs -P $MAX_JOBS -I {} bash -c 'if python ../../scripts/map_QM_to_CG.py {} ../refer_CG 2>&1; then
        echo "[succeed] {}"
    else
        echo "[failed] {}"
        echo "{}" >> "$0"
    fi
' "$FAILED_FILE"
```

4. Enter data_set/refer_CG
cp ../refer_QM/task_dirs.list all_data_dirs.list 
python ../../script/solve_CG.py  =>  generate all_energy_files.txt and all_force_files.txt

5. Enter main directoryrectory
python optimize_main.py 64 1
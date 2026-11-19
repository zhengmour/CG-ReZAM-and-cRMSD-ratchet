"""N05 topology and N04 history analysis; inputs and outputs are explicit."""
import argparse
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
from pirmsd_io import read_pirmsd_history,write_history_csv,write_cycle_metrics


@contextmanager
def output_directory(path):
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True);old=Path.cwd()
    try:os.chdir(path);yield
    finally:os.chdir(old)


def main(file_type,n_loop,traj_path='./',output=None,dt_fs=None,metrics_only=False,type_map=None):
    if n_loop<1:raise ValueError('workers must be positive')
    if dt_fs is not None and (not np.isfinite(dt_fs) or dt_fs<=0):raise ValueError('dt-fs must be positive')
    if type_map is not None:
        import json
        type_map=json.loads(Path(type_map).resolve().read_text())
        if set(type_map)!={'Si','OB','C','N','Na'}:raise ValueError('type-map needs Si, OB, C, N, Na')
        assigned=[]
        for value in type_map.values():
            values=value if isinstance(value,list) else [value]
            if not values or any(not isinstance(v,int) or v<1 for v in values):raise ValueError('Atom types must be positive integers')
            assigned.extend(values)
        if len(set(assigned))!=len(assigned):raise ValueError('Atom type selections must not overlap')
    root=Path(traj_path).resolve();out=Path(output).resolve() if output else root/'analysis'
    with output_directory(out):
        return analyze(file_type,n_loop,root,dt_fs,metrics_only,type_map)


def analyze(file_type,n_loop,root,dt_fs,metrics_only,type_map):
    import topology_plot as plots
    records=read_pirmsd_history(root)
    stage_count = write_cycle_metrics(root, 'Analysis__pirmsd_stages.csv')
    if records:
        write_history_csv(records,'Analysis__pirmsd.csv');plots.plot_pirmsd_records(records)
        print('Read',len(records),'PIRMSD observations; units and reference epochs retained.')
    if metrics_only:
        if not records and not stage_count:raise FileNotFoundError('No PIRMSD history records')
        return
    from multiprocessing import Pool
    from topology_io import read_gro_files,read_xtc_files,read_data_files,read_dump_files,read_rmsd_dump_files
    from lammps_io import TYPE_MAP
    from topology_utils import process_chunk_build_graphs,analyze_graph_frame,analyze_graph_pair_reaction
    from topology_analysis import aggregate_topology_results,aggregate_topology_reactions_results
    from topology_constants import MAX_LENGTH,REACTIONS
    types=type_map or TYPE_MAP
    if file_type=='gro':topologies=read_gro_files(root,n_loop)
    elif file_type=='xtc':topologies=read_xtc_files(root,n_loop)
    elif file_type in ('lmp','data'):topologies=read_data_files(root,n_loop,types)
    elif file_type in ('dump','lammpstrj'):topologies=read_dump_files(root,n_loop,types,dt_fs)
    elif file_type=='rmsd_lammpstrj':topologies=read_rmsd_dump_files(root,n_loop,types,dt_fs)
    else:raise ValueError('Unsupported file type: '+file_type)
    if not topologies:raise ValueError('No trajectory frames')
    for _,static in topologies:
        if not len(static['idx_Si']) or not len(static['idx_OB']):
            raise ValueError('Topology analysis needs Si and SP/OB sites; verify atom selections')
    print('Loaded',len(topologies),'frames.')
    import csv
    with open('Analysis__frames.csv','w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','segment','md_step','coordinate','axis_unit','source'])
        for index,(frame,_) in enumerate(topologies):writer.writerow([index,frame.get('segment',''),frame.get('md_step',''),frame['time'],frame.get('axis_unit','ps'),frame.get('source','')])
    unit=topologies[0][0].get('axis_unit','ps');factor=.001 if unit=='ps' else 1.
    axis_name='Time_ns' if unit=='ps' else ('MD_step' if unit=='step' else 'Frame')
    plots.TIME_LABEL='Time / ns (segment-local)' if unit=='ps' and file_type=='rmsd_lammpstrj' else ('Time / ns' if unit=='ps' else axis_name)
    # Bond history must not cross independent N04 segments. A finite chunk
    # overlap cannot guarantee exact hysteresis, so parallelize whole segments.
    segments=[]
    for topology in topologies:
        key=topology[0].get('segment','trajectory')
        if not segments or segments[-1][0]!=key:segments.append((key,[]))
        segments[-1][1].append(topology)
    with (Pool(n_loop) if n_loop > 1 else nullcontext()) as pool:
        dispatch = pool.map if pool else map
        graph_parts=dispatch(process_chunk_build_graphs,[frames for _,frames in segments])
        graphs=[g for part in graph_parts for g in part]
        results, cycles = zip(*dispatch(analyze_graph_frame,
                                        [(i, graphs[i], t, True) for i, t in enumerate(topologies)]))
    times,qns,chains,rings,clusters=aggregate_topology_results([result for _,result in results],topologies)
    times=np.asarray(times)*factor
    def save(name,columns,header):np.savetxt('Analysis__'+name+'.csv',np.column_stack(columns),delimiter=',',header=axis_name+','+header,comments='')
    save('Qns',[times,qns],','.join(['Q'+str(i) for i in range(qns.shape[1]-1)]+['C']))
    for name,data,prefix in zip(['Linear','Branch','Cyclic'],chains,['L','B','C']):save(name,[times,data],','.join(prefix+str(i+1) for i in range(MAX_LENGTH)))
    save('Rings',[times,rings],','.join('R'+str(i+1) for i in range(MAX_LENGTH)))
    save('Cluster',[times,clusters],'Maximum bonded cluster size (>=50 Si),Mean bonded cluster size (>=50 Si),Number of bonded clusters (>=50 Si)')
    plots.plot_qns(times,qns);plots.plot_c_qn(qns);plots.plot_chains(times,chains)
    plots.plot_rings(times,rings);plots.plot_Nrings(times,rings);plots.plot_clusters(times,clusters)
    # Report counts per adjacent saved-frame interval, not physical reaction
    # rates. Never bridge gaps between independently sampled PIRMSD segments.
    pairs=[(i,i+1) for i in range(len(topologies)-1) if topologies[i][0].get('segment')==topologies[i+1][0].get('segment')]
    if pairs:
        with (Pool(n_loop) if n_loop > 1 else nullcontext()) as pool:
            dispatch = pool.map if pool else map
            result = list(dispatch(analyze_graph_pair_reaction,
                                   [(k, graphs[i], graphs[j], topologies[i], topologies[j], cycles[i], cycles[j])
                                    for k, (i, j) in enumerate(pairs)]))
        reaction_times,counts=aggregate_topology_reactions_results(result);reaction_times*=factor
        save('reactions',[reaction_times]+[counts[key] for key in REACTIONS],','.join(REACTIONS))
        plots.plot_reactions(reaction_times,counts)
    else:print('No within-segment frame pairs; reaction analysis skipped.')
    print('Analysis complete; output:',Path.cwd())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file_type',choices=['gro','xtc','lmp','data','dump','lammpstrj','rmsd_lammpstrj'])
    parser.add_argument('workers',type=int,nargs='?',default=1)
    parser.add_argument('trajectory',nargs='?',default='.')
    parser.add_argument('--output');parser.add_argument('--dt-fs',type=float)
    parser.add_argument('--type-map',help='JSON mapping Si/OB/C/N/Na to integer types or lists')
    parser.add_argument('--metrics-only',action='store_true',help='Read histories only, without trajectory/topology analysis')
    args=parser.parse_args();main(args.file_type,args.workers,args.trajectory,args.output,args.dt_fs,args.metrics_only,args.type_map)

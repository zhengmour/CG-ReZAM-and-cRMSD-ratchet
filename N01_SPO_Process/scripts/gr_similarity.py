"""Pair-name-matched RDF objective, fixed across all scanned parameters."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workflow import load_workflow
from spo_loss import rdf_distance


def score_rdfs(current_dir,reference_dir,settings):
    scores={};proposals={}
    for pair in settings['rdf_pairs']:
        name='rdf_'+pair+'.csv'
        current=np.loadtxt(Path(current_dir)/name,delimiter=',');reference=np.loadtxt(Path(reference_dir)/name,delimiter=',')
        scores[pair]=rdf_distance(current,reference,settings['rdf_lower_angstrom'],settings['rdf_cutoff_angstrom'],settings['rdf_metric'])
        # Peak information proposes a trial; it must never overwrite a fitted value without evaluation.
        indices,_=find_peaks(reference[:,1],prominence=max(.02,.05*np.ptp(reference[:,1])))
        indices=[i for i in indices if settings['rdf_lower_angstrom']<=reference[i,0]<=settings['rdf_cutoff_angstrom']]
        proposals[pair+'_1']=float(reference[indices[0],0]/2**(1/6)) if indices else None
    return dict(pairs=scores,total=float(sum(scores.values())),metric=settings['rdf_metric']),proposals


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--current',required=True,type=Path)
    p.add_argument('--reference',type=Path);args=p.parse_args();cfg=load_workflow(args.config)
    result,proposals=score_rdfs(args.current,args.reference or cfg['paths']['rdf_reference'],cfg['objective'])
    (args.current/'rdf_loss.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.current/'score.txt').write_text(''.join('rdf_'+k+' '+str(v)+'\n' for k,v in result['pairs'].items())+'total '+str(result['total'])+'\n')
    (args.current/'rdf_first_sigma_file.json').write_text(json.dumps(proposals,indent=2)+'\n')
if __name__=='__main__':main()

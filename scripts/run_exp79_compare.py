import json
from pathlib import Path
from backend.app.services.lifecycle_model_comparison_service import retrain_and_compare
WINDOWS=((2001,2019),(2001,2021)); REQUIRED=("production_cost_mae","experiment_cost_mae","cost_improvement_percentage","production_delay_mae","experiment_delay_mae","delay_improvement_percentage","verdict")
def main():
 out=Path("audit_outputs/exp79"); out.mkdir(parents=True,exist_ok=True); results={}
 for start,end in WINDOWS:
  key=f"{start}_{end}"; payload=retrain_and_compare(start,end,"exp_79"); overall=payload.get("overall_comparison") or {}; missing=[k for k in REQUIRED if overall.get(k) is None]
  if missing: raise RuntimeError(f"{key} incomplete: {missing}")
  (out/f"{key}.json").write_text(json.dumps(payload,indent=2,default=str,allow_nan=False)+"\n"); results[key]=overall
  print(f"{key}: cost {overall['production_cost_mae']} -> {overall['experiment_cost_mae']} ({overall['cost_improvement_percentage']}%); delay {overall['production_delay_mae']} -> {overall['experiment_delay_mae']} ({overall['delay_improvement_percentage']}%); verdict={overall['verdict']}")
 (out/"summary.json").write_text(json.dumps({"experiment_id":"exp_79","windows":results,"promotion_allowed":False},indent=2,allow_nan=False)+"\n")
if __name__=="__main__": main()

import numpy as np,pandas as pd
from backend.app.ml.experiments.exp139_physical_feasibility import cost_lower_bound,delay_lower_bound,project_lower_bound

def test_cost_lower_bound_math():
    f=pd.DataFrame({"approved_cost_cr":[100.,200.],"cumulative_expenditure_cr":[120.,150.]})
    assert np.allclose(cost_lower_bound(f),[20.,-25.])

def test_delay_lower_bound_math():
    f=pd.DataFrame({"snapshot_date":["2022-02-01","2020-01-01"],"planned_completion_date":["2022-01-01","2021-01-01"]})
    assert delay_lower_bound(f).tolist()==[31,0]

def test_projection_never_lowers_and_can_be_disabled():
    p=np.array([10.,30.]);lb=pd.Series([20.,25.]);out,mask=project_lower_bound(p,lb,True)
    assert np.all(out>=p) and out.tolist()==[20.,30.] and mask.tolist()==[True,False]
    unchanged,_=project_lower_bound(p,lb,False);assert np.array_equal(unchanged,p)

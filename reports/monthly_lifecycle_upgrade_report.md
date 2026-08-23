# PAIMANA monthly lifecycle forecasting upgrade

## Data

- Official reports discovered: 291
- Financial years: 24 (2001-02, 2002-03, 2003-04, 2004-05, 2005-06, 2006-07, 2007-08, 2008-09, 2009-10, 2010-11, 2011-12, 2012-13, 2013-14, 2014-15, 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)
- Known missing months: April 2005-06, January 2008-09
- Reports parsed with project rows: 274
- Reports downloaded/cached: 290; download failures: 1
- Downloaded reports with no recognized project rows: 16
- Monthly observations: 195934
- Unique reported project codes: 4499
- Canonical trajectories generated: 10284
- Parser row-success rate among downloaded reports: 94.5% (summary-only/unrecognized reports remain audited)
- Parser coverage: {"classic-code-v3": 143, "legacy-sector-v1": 98, "recent-project-list-v3": 9, "redesigned-code-v1": 24, "unrecognized": 16}
- Observations by financial year: {"2001-02": 2134, "2002-03": 2983, "2003-04": 3172, "2004-05": 3479, "2005-06": 3556, "2006-07": 5170, "2007-08": 5446, "2008-09": 5630, "2009-10": 6718, "2010-11": 6253, "2011-12": 5949, "2012-13": 7910, "2013-14": 7572, "2014-15": 8128, "2015-16": 8112, "2016-17": 4939, "2017-18": 10191, "2018-19": 11961, "2019-20": 16912, "2020-21": 13629, "2021-22": 9771, "2022-23": 12078, "2023-24": 18803, "2024-25": 15438}

## Identity

- Snapshot rows audited: 195934
- Identity-verified rows: 95738
- Ambiguous exact-name rows excluded: 140
- Identity methods: {"ambiguous_exact_name": 140, "exact_name_and_approved_cost": 5217, "exact_name_cost_mismatch": 277, "exact_official_project_id": 90521, "unresolved": 99779}
- Canonical completed outcomes: 4797 (2292 with unique official IDs)

No fuzzy name matching is used. Unverified or ambiguous rows are excluded from supervised training.

## Window 2001_2015

### Baseline versus lifecycle

Training: 7306 snapshots / 757 projects. Test: 12640 snapshots / 821 projects.

Selected regressors: cost=xgboost; delay=lightgbm. Risk uses the documented Random Forest classifier.

| Metric | Five-feature baseline | Monthly lifecycle |
|---|---:|---:|
| Cost MAE | 49.676 | 41.484 |
| Cost RMSE | 82.99 | 73.092 |
| Cost R2 | 0.012 | 0.2336 |
| Delay MAE | 851.845 | 490.459 |
| Delay RMSE | 1259.429 | 780.728 |
| Delay R2 | -0.1145 | 0.5717 |
| Risk accuracy | 0.2108 | 0.5717 |
| Risk macro F1 | 0.1649 | 0.4444 |

Primary MAE improvement: cost 16.5%; delay 42.4%.

### Feature audit

Retained (25): approved_cost_cr, cumulative_expenditure_cr, expenditure_ratio, schedule_slippage_days, schedule_slippage_ratio, elapsed_duration_days, planned_duration_days, duration_ratio, expected_progress_percentage, revised_cost_cr, cost_escalation_percentage, sector, project_size_category, implementing_agency, cost_growth_velocity_3m, cost_growth_velocity_6m, cost_acceleration, sector_average_delay, sector_average_cost_overrun, sector_delay_rate, sector_cost_overrun_rate, agency_average_delay, agency_average_cost_overrun, agency_delay_rate, agency_cost_overrun_rate.

Rejected (7): physical_progress (availability below 10.0%); progress_deviation (availability below 10.0%); current_schedule_status (availability below 10.0%); ministry (availability below 10.0%); progress_velocity_3m (availability below 10.0%); progress_velocity_6m (availability below 10.0%); progress_acceleration (availability below 10.0%).

| Feature | Available | Missing | Years | Projects | As-of safe | Decision | Reason |
|---|---:|---:|---:|---:|---|---|---|
| approved_cost_cr | 100.0% | 0.0% | 15 | 757 | yes | keep | observed and variable in the training window |
| cumulative_expenditure_cr | 100.0% | 0.0% | 15 | 757 | yes | keep | observed and variable in the training window |
| expenditure_ratio | 99.97% | 0.03% | 15 | 757 | yes | keep | observed and variable in the training window |
| physical_progress | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| schedule_slippage_days | 95.91% | 4.09% | 15 | 751 | yes | keep | observed and variable in the training window |
| schedule_slippage_ratio | 95.51% | 4.49% | 15 | 747 | yes | keep | observed and variable in the training window |
| elapsed_duration_days | 100.0% | 0.0% | 15 | 757 | yes | keep | observed and variable in the training window |
| planned_duration_days | 95.91% | 4.09% | 15 | 751 | yes | keep | observed and variable in the training window |
| duration_ratio | 95.51% | 4.49% | 15 | 747 | yes | keep | observed and variable in the training window |
| expected_progress_percentage | 95.51% | 4.49% | 15 | 747 | yes | keep | observed and variable in the training window |
| progress_deviation | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| revised_cost_cr | 99.44% | 0.56% | 15 | 747 | yes | keep | observed and variable in the training window |
| cost_escalation_percentage | 99.41% | 0.59% | 15 | 747 | yes | keep | observed and variable in the training window |
| current_schedule_status | 9.21% | 90.79% | 9 | 113 | yes | remove | availability below 10.0% |
| sector | 97.44% | 2.56% | 15 | 755 | yes | keep | observed and variable in the training window |
| project_size_category | 100.0% | 0.0% | 15 | 757 | yes | keep | observed and variable in the training window |
| implementing_agency | 73.51% | 26.49% | 15 | 621 | yes | keep | observed and variable in the training window |
| ministry | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| cost_growth_velocity_3m | 87.53% | 12.47% | 15 | 690 | yes | keep | observed and variable in the training window |
| cost_growth_velocity_6m | 92.01% | 7.99% | 15 | 692 | yes | keep | observed and variable in the training window |
| cost_acceleration | 87.53% | 12.47% | 15 | 690 | yes | keep | observed and variable in the training window |
| progress_velocity_3m | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| progress_velocity_6m | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| progress_acceleration | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| sector_average_delay | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| sector_average_cost_overrun | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| sector_delay_rate | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| sector_cost_overrun_rate | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| agency_average_delay | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| agency_average_cost_overrun | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| agency_delay_rate | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |
| agency_cost_overrun_rate | 99.86% | 0.14% | 15 | 752 | yes | keep | observed and variable in the training window |

### Lifecycle-stage evaluation

```json
{
  "early": {
    "available": true,
    "cost": {
      "MAE": 41.835,
      "RMSE": 67.171,
      "R2": -0.1101,
      "MAPE": 171.323,
      "rows": 474,
      "unique_projects": 200
    },
    "delay": {
      "MAE": 407.191,
      "RMSE": 604.165,
      "R2": -0.0356,
      "MAPE": 92.903,
      "rows": 474,
      "unique_projects": 200
    },
    "risk": {
      "accuracy": 0.4087,
      "macro_precision": 0.5027,
      "macro_recall": 0.2755,
      "macro_f1": 0.1932,
      "confusion_matrix": [
        [
          3.2638,
          0.0,
          0.0237,
          0.0
        ],
        [
          1.4705,
          0.1157,
          0.0934,
          0.0
        ],
        [
          1.3899,
          0.054,
          0.0502,
          0.0
        ],
        [
          1.7489,
          0.1343,
          0.0656,
          0.0129
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  },
  "mid": {
    "available": true,
    "cost": {
      "MAE": 38.385,
      "RMSE": 60.159,
      "R2": -0.1861,
      "MAPE": 187.208,
      "rows": 1208,
      "unique_projects": 389
    },
    "delay": {
      "MAE": 401.074,
      "RMSE": 598.747,
      "R2": -0.0687,
      "MAPE": 99.132,
      "rows": 1208,
      "unique_projects": 389
    },
    "risk": {
      "accuracy": 0.3304,
      "macro_precision": 0.4308,
      "macro_recall": 0.2714,
      "macro_f1": 0.1822,
      "confusion_matrix": [
        [
          8.2283,
          0.1538,
          0.2986,
          0.0078
        ],
        [
          5.3142,
          0.378,
          0.2344,
          0.0
        ],
        [
          5.0137,
          0.3592,
          0.2799,
          0.0
        ],
        [
          5.547,
          1.0143,
          0.4346,
          0.1823
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  },
  "late": {
    "available": true,
    "cost": {
      "MAE": 35.394,
      "RMSE": 52.066,
      "R2": 0.0245,
      "MAPE": 223.285,
      "rows": 1741,
      "unique_projects": 512
    },
    "delay": {
      "MAE": 450.739,
      "RMSE": 685.259,
      "R2": -0.105,
      "MAPE": 110.889,
      "rows": 1741,
      "unique_projects": 512
    },
    "risk": {
      "accuracy": 0.2489,
      "macro_precision": 0.2923,
      "macro_recall": 0.2868,
      "macro_f1": 0.1778,
      "confusion_matrix": [
        [
          8.8961,
          0.25,
          0.1206,
          0.1839
        ],
        [
          10.1951,
          0.9121,
          0.601,
          0.1683
        ],
        [
          8.3496,
          0.7522,
          0.802,
          0.3982
        ],
        [
          8.722,
          2.3326,
          2.0808,
          0.709
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  },
  "very_late": {
    "available": true,
    "cost": {
      "MAE": 40.353,
      "RMSE": 72.182,
      "R2": 0.2867,
      "MAPE": 199.092,
      "rows": 8615,
      "unique_projects": 736
    },
    "delay": {
      "MAE": 488.048,
      "RMSE": 747.115,
      "R2": 0.6248,
      "MAPE": 42.179,
      "rows": 8615,
      "unique_projects": 736
    },
    "risk": {
      "accuracy": 0.6947,
      "macro_precision": 0.4493,
      "macro_recall": 0.5737,
      "macro_f1": 0.4736,
      "confusion_matrix": [
        [
          2.6579,
          0.8469,
          0.129,
          0.0926
        ],
        [
          4.3811,
          7.9217,
          6.0813,
          4.4361
        ],
        [
          3.2442,
          4.4858,
          12.657,
          8.7446
        ],
        [
          3.1626,
          10.2362,
          15.8867,
          117.1957
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  }
}
```

### Ablations

```json
{
  "without_revised_cost": {
    "features": [
      "approved_cost_cr",
      "sector_average_delay",
      "sector_average_cost_overrun",
      "sector",
      "project_size_category",
      "cumulative_expenditure_cr",
      "expenditure_ratio",
      "schedule_slippage_days",
      "schedule_slippage_ratio",
      "elapsed_duration_days",
      "planned_duration_days",
      "duration_ratio",
      "expected_progress_percentage",
      "implementing_agency",
      "cost_growth_velocity_3m",
      "cost_growth_velocity_6m",
      "cost_acceleration",
      "sector_delay_rate",
      "sector_cost_overrun_rate",
      "agency_average_delay",
      "agency_average_cost_overrun",
      "agency_delay_rate",
      "agency_cost_overrun_rate"
    ],
    "selected_algorithms": {
      "cost": "xgboost",
      "delay": "lightgbm"
    },
    "internal_algorithm_comparisons": {
      "cost": [
        {
          "algorithm": "lightgbm",
          "MAE": 41.195,
          "RMSE": 89.365,
          "R2": 0.2027,
          "MAPE": 181.532,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "xgboost",
          "MAE": 40.354,
          "RMSE": 90.614,
          "R2": 0.1802,
          "MAPE": 170.254,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "extra_trees",
          "MAE": 42.653,
          "RMSE": 85.409,
          "R2": 0.2717,
          "MAPE": 173.96,
          "rows": 1096,
          "unique_projects": 71
        }
      ],
      "delay": [
        {
          "algorithm": "lightgbm",
          "MAE": 352.052,
          "RMSE": 564.184,
          "R2": 0.5669,
          "MAPE": 60.473,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "xgboost",
          "MAE": 376.8,
          "RMSE": 577.149,
          "R2": 0.5468,
          "MAPE": 65.493,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "extra_trees",
          "MAE": 368.479,
          "RMSE": 551.81,
          "R2": 0.5857,
          "MAPE": 75.95,
          "rows": 1096,
          "unique_projects": 71
        }
      ]
    },
    "metrics": {
      "cost": {
        "MAE": 42.139,
        "RMSE": 73.92,
        "R2": 0.2162,
        "MAPE": 190.358,
        "rows": 12640,
        "unique_projects": 821
      },
      "delay": {
        "MAE": 494.313,
        "RMSE": 781.736,
        "R2": 0.5706,
        "MAPE": 61.991,
        "rows": 12640,
        "unique_projects": 821
      },
      "risk": {
        "accuracy": 0.5709,
        "macro_precision": 0.4498,
        "macro_recall": 0.5214,
        "macro_f1": 0.4405,
        "confusion_matrix": [
          [
            22.8625,
            1.9886,
            0.622,
            0.3816
          ],
          [
            22.4214,
            10.3424,
            6.8095,
            4.2546
          ],
          [
            18.142,
            7.6818,
            13.4393,
            9.8204
          ],
          [
            19.9466,
            13.4742,
            20.7744,
            121.4192
          ]
        ],
        "labels": [
          "LOW",
          "MEDIUM",
          "HIGH",
          "CRITICAL"
        ]
      }
    },
    "lifecycle_stages": {
      "early": {
        "available": true,
        "cost": {
          "MAE": 41.567,
          "RMSE": 66.774,
          "R2": -0.097,
          "MAPE": 171.751,
          "rows": 474,
          "unique_projects": 200
        },
        "delay": {
          "MAE": 414.383,
          "RMSE": 611.109,
          "R2": -0.0596,
          "MAPE": 91.108,
          "rows": 474,
          "unique_projects": 200
        },
        "risk": {
          "accuracy": 0.3992,
          "macro_precision": 0.2974,
          "macro_recall": 0.263,
          "macro_f1": 0.1713,
          "confusion_matrix": [
            [
              3.2588,
              0.0109,
              0.0058,
              0.012
            ],
            [
              1.5253,
              0.0835,
              0.0709,
              0.0
            ],
            [
              1.2968,
              0.1903,
              0.0069,
              0.0
            ],
            [
              1.7779,
              0.149,
              0.0218,
              0.0129
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "mid": {
        "available": true,
        "cost": {
          "MAE": 36.975,
          "RMSE": 56.66,
          "R2": -0.0522,
          "MAPE": 186.123,
          "rows": 1208,
          "unique_projects": 389
        },
        "delay": {
          "MAE": 406.336,
          "RMSE": 600.172,
          "R2": -0.0738,
          "MAPE": 104.425,
          "rows": 1208,
          "unique_projects": 389
        },
        "risk": {
          "accuracy": 0.3292,
          "macro_precision": 0.4006,
          "macro_recall": 0.2692,
          "macro_f1": 0.1779,
          "confusion_matrix": [
            [
              8.2617,
              0.1647,
              0.2302,
              0.0319
            ],
            [
              5.4399,
              0.3521,
              0.1346,
              0.0
            ],
            [
              4.9711,
              0.4692,
              0.2124,
              0.0
            ],
            [
              5.4659,
              1.1486,
              0.3547,
              0.2089
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "late": {
        "available": true,
        "cost": {
          "MAE": 36.651,
          "RMSE": 55.206,
          "R2": -0.0967,
          "MAPE": 224.265,
          "rows": 1741,
          "unique_projects": 512
        },
        "delay": {
          "MAE": 448.742,
          "RMSE": 683.973,
          "R2": -0.1008,
          "MAPE": 115.823,
          "rows": 1741,
          "unique_projects": 512
        },
        "risk": {
          "accuracy": 0.2403,
          "macro_precision": 0.2772,
          "macro_recall": 0.2783,
          "macro_f1": 0.1643,
          "confusion_matrix": [
            [
              8.9294,
              0.25,
              0.0873,
              0.1839
            ],
            [
              10.6851,
              0.5859,
              0.4371,
              0.1683
            ],
            [
              8.3593,
              0.8734,
              0.6746,
              0.3948
            ],
            [
              8.8369,
              2.5525,
              1.7158,
              0.7392
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "very_late": {
        "available": true,
        "cost": {
          "MAE": 41.359,
          "RMSE": 73.694,
          "R2": 0.2565,
          "MAPE": 183.098,
          "rows": 8615,
          "unique_projects": 736
        },
        "delay": {
          "MAE": 493.351,
          "RMSE": 749.202,
          "R2": 0.6227,
          "MAPE": 44.297,
          "rows": 8615,
          "unique_projects": 736
        },
        "risk": {
          "accuracy": 0.6968,
          "macro_precision": 0.4503,
          "macro_recall": 0.5584,
          "macro_f1": 0.4678,
          "confusion_matrix": [
            [
              2.4055,
              1.1326,
              0.1474,
              0.0409
            ],
            [
              4.6511,
              8.276,
              5.8708,
              4.0224
            ],
            [
              3.4845,
              4.3463,
              12.2393,
              9.0613
            ],
            [
              3.698,
              7.6025,
              17.2429,
              117.9378
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      }
    }
  },
  "snapshot_only": {
    "features": [
      "approved_cost_cr",
      "sector_average_delay",
      "sector_average_cost_overrun",
      "sector",
      "project_size_category",
      "cumulative_expenditure_cr",
      "expenditure_ratio",
      "schedule_slippage_days",
      "schedule_slippage_ratio",
      "elapsed_duration_days",
      "planned_duration_days",
      "duration_ratio",
      "expected_progress_percentage",
      "revised_cost_cr",
      "cost_escalation_percentage",
      "implementing_agency",
      "sector_delay_rate",
      "sector_cost_overrun_rate",
      "agency_average_delay",
      "agency_average_cost_overrun",
      "agency_delay_rate",
      "agency_cost_overrun_rate"
    ],
    "selected_algorithms": {
      "cost": "xgboost",
      "delay": "lightgbm"
    },
    "internal_algorithm_comparisons": {
      "cost": [
        {
          "algorithm": "lightgbm",
          "MAE": 40.74,
          "RMSE": 87.576,
          "R2": 0.2343,
          "MAPE": 170.29,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "xgboost",
          "MAE": 39.284,
          "RMSE": 84.68,
          "R2": 0.2841,
          "MAPE": 167.167,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "extra_trees",
          "MAE": 42.124,
          "RMSE": 82.008,
          "R2": 0.3286,
          "MAPE": 175.057,
          "rows": 1096,
          "unique_projects": 71
        }
      ],
      "delay": [
        {
          "algorithm": "lightgbm",
          "MAE": 344.685,
          "RMSE": 558.209,
          "R2": 0.576,
          "MAPE": 58.029,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "xgboost",
          "MAE": 369.963,
          "RMSE": 566.205,
          "R2": 0.5638,
          "MAPE": 63.781,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "extra_trees",
          "MAE": 363.426,
          "RMSE": 539.465,
          "R2": 0.604,
          "MAPE": 75.39,
          "rows": 1096,
          "unique_projects": 71
        }
      ]
    },
    "metrics": {
      "cost": {
        "MAE": 41.666,
        "RMSE": 74.034,
        "R2": 0.2137,
        "MAPE": 202.846,
        "rows": 12640,
        "unique_projects": 821
      },
      "delay": {
        "MAE": 489.852,
        "RMSE": 780.152,
        "R2": 0.5723,
        "MAPE": 59.984,
        "rows": 12640,
        "unique_projects": 821
      },
      "risk": {
        "accuracy": 0.5639,
        "macro_precision": 0.4468,
        "macro_recall": 0.5163,
        "macro_f1": 0.4388,
        "confusion_matrix": [
          [
            22.1278,
            2.0475,
            0.8323,
            0.8471
          ],
          [
            22.1367,
            10.0108,
            6.8013,
            4.8791
          ],
          [
            17.626,
            7.0121,
            14.8792,
            9.5662
          ],
          [
            19.5938,
            14.0056,
            23.0189,
            118.9961
          ]
        ],
        "labels": [
          "LOW",
          "MEDIUM",
          "HIGH",
          "CRITICAL"
        ]
      }
    },
    "lifecycle_stages": {
      "early": {
        "available": true,
        "cost": {
          "MAE": 41.358,
          "RMSE": 66.826,
          "R2": -0.0987,
          "MAPE": 166.284,
          "rows": 474,
          "unique_projects": 200
        },
        "delay": {
          "MAE": 406.708,
          "RMSE": 604.392,
          "R2": -0.0364,
          "MAPE": 90.334,
          "rows": 474,
          "unique_projects": 200
        },
        "risk": {
          "accuracy": 0.3904,
          "macro_precision": 0.3635,
          "macro_recall": 0.251,
          "macro_f1": 0.1475,
          "confusion_matrix": [
            [
              3.2754,
              0.0,
              0.012,
              0.0
            ],
            [
              1.6088,
              0.0,
              0.0709,
              0.0
            ],
            [
              1.4331,
              0.054,
              0.0069,
              0.0
            ],
            [
              1.7785,
              0.1267,
              0.0506,
              0.0058
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "mid": {
        "available": true,
        "cost": {
          "MAE": 38.2,
          "RMSE": 60.16,
          "R2": -0.1862,
          "MAPE": 182.734,
          "rows": 1208,
          "unique_projects": 389
        },
        "delay": {
          "MAE": 403.397,
          "RMSE": 600.157,
          "R2": -0.0738,
          "MAPE": 100.747,
          "rows": 1208,
          "unique_projects": 389
        },
        "risk": {
          "accuracy": 0.3216,
          "macro_precision": 0.3086,
          "macro_recall": 0.258,
          "macro_f1": 0.1529,
          "confusion_matrix": [
            [
              8.429,
              0.0,
              0.1685,
              0.091
            ],
            [
              5.7068,
              0.0613,
              0.1584,
              0.0
            ],
            [
              5.2653,
              0.2582,
              0.1292,
              0.0
            ],
            [
              5.6752,
              0.9167,
              0.3797,
              0.2065
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "late": {
        "available": true,
        "cost": {
          "MAE": 35.151,
          "RMSE": 51.633,
          "R2": 0.0407,
          "MAPE": 220.658,
          "rows": 1741,
          "unique_projects": 512
        },
        "delay": {
          "MAE": 449.797,
          "RMSE": 684.512,
          "R2": -0.1026,
          "MAPE": 112.631,
          "rows": 1741,
          "unique_projects": 512
        },
        "risk": {
          "accuracy": 0.2481,
          "macro_precision": 0.2906,
          "macro_recall": 0.2855,
          "macro_f1": 0.193,
          "confusion_matrix": [
            [
              8.228,
              0.2656,
              0.4065,
              0.5505
            ],
            [
              10.7337,
              0.483,
              0.4914,
              0.1683
            ],
            [
              7.3843,
              0.6597,
              1.8112,
              0.4468
            ],
            [
              8.9115,
              2.4498,
              1.7221,
              0.761
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "very_late": {
        "available": true,
        "cost": {
          "MAE": 40.783,
          "RMSE": 73.511,
          "R2": 0.2602,
          "MAPE": 201.941,
          "rows": 8615,
          "unique_projects": 736
        },
        "delay": {
          "MAE": 487.137,
          "RMSE": 746.623,
          "R2": 0.6253,
          "MAPE": 42.398,
          "rows": 8615,
          "unique_projects": 736
        },
        "risk": {
          "accuracy": 0.6848,
          "macro_precision": 0.4412,
          "macro_recall": 0.5426,
          "macro_f1": 0.4606,
          "confusion_matrix": [
            [
              2.1883,
              1.3711,
              0.0745,
              0.0926
            ],
            [
              4.0786,
              8.3076,
              5.8364,
              4.5978
            ],
            [
              3.5129,
              4.2268,
              12.5636,
              8.8282
            ],
            [
              3.1429,
              8.7019,
              19.249,
              115.3874
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      }
    }
  },
  "without_agency_priors": {
    "features": [
      "approved_cost_cr",
      "sector_average_delay",
      "sector_average_cost_overrun",
      "sector",
      "project_size_category",
      "cumulative_expenditure_cr",
      "expenditure_ratio",
      "schedule_slippage_days",
      "schedule_slippage_ratio",
      "elapsed_duration_days",
      "planned_duration_days",
      "duration_ratio",
      "expected_progress_percentage",
      "revised_cost_cr",
      "cost_escalation_percentage",
      "implementing_agency",
      "cost_growth_velocity_3m",
      "cost_growth_velocity_6m",
      "cost_acceleration",
      "sector_delay_rate",
      "sector_cost_overrun_rate"
    ],
    "selected_algorithms": {
      "cost": "xgboost",
      "delay": "lightgbm"
    },
    "internal_algorithm_comparisons": {
      "cost": [
        {
          "algorithm": "lightgbm",
          "MAE": 41.186,
          "RMSE": 87.567,
          "R2": 0.2344,
          "MAPE": 175.463,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "xgboost",
          "MAE": 40.128,
          "RMSE": 83.797,
          "R2": 0.2989,
          "MAPE": 170.315,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "extra_trees",
          "MAE": 42.39,
          "RMSE": 81.172,
          "R2": 0.3422,
          "MAPE": 176.444,
          "rows": 1096,
          "unique_projects": 71
        }
      ],
      "delay": [
        {
          "algorithm": "lightgbm",
          "MAE": 346.041,
          "RMSE": 555.881,
          "R2": 0.5795,
          "MAPE": 59.671,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "xgboost",
          "MAE": 370.816,
          "RMSE": 567.224,
          "R2": 0.5622,
          "MAPE": 66.398,
          "rows": 1096,
          "unique_projects": 71
        },
        {
          "algorithm": "extra_trees",
          "MAE": 366.0,
          "RMSE": 548.462,
          "R2": 0.5907,
          "MAPE": 74.825,
          "rows": 1096,
          "unique_projects": 71
        }
      ]
    },
    "metrics": {
      "cost": {
        "MAE": 40.835,
        "RMSE": 73.332,
        "R2": 0.2286,
        "MAPE": 199.339,
        "rows": 12640,
        "unique_projects": 821
      },
      "delay": {
        "MAE": 491.432,
        "RMSE": 780.754,
        "R2": 0.5717,
        "MAPE": 61.234,
        "rows": 12640,
        "unique_projects": 821
      },
      "risk": {
        "accuracy": 0.5357,
        "macro_precision": 0.4391,
        "macro_recall": 0.5195,
        "macro_f1": 0.4319,
        "confusion_matrix": [
          [
            23.1742,
            1.8358,
            0.5266,
            0.3181
          ],
          [
            22.0529,
            11.6196,
            6.9706,
            3.1847
          ],
          [
            17.2971,
            9.5889,
            14.7683,
            7.4291
          ],
          [
            20.216,
            20.1034,
            27.1648,
            108.1302
          ]
        ],
        "labels": [
          "LOW",
          "MEDIUM",
          "HIGH",
          "CRITICAL"
        ]
      }
    },
    "lifecycle_stages": {
      "early": {
        "available": true,
        "cost": {
          "MAE": 41.228,
          "RMSE": 66.314,
          "R2": -0.0819,
          "MAPE": 174.013,
          "rows": 474,
          "unique_projects": 200
        },
        "delay": {
          "MAE": 411.951,
          "RMSE": 608.471,
          "R2": -0.0504,
          "MAPE": 95.346,
          "rows": 474,
          "unique_projects": 200
        },
        "risk": {
          "accuracy": 0.4034,
          "macro_precision": 0.3482,
          "macro_recall": 0.2672,
          "macro_f1": 0.1767,
          "confusion_matrix": [
            [
              3.2754,
              0.0,
              0.0,
              0.012
            ],
            [
              1.5253,
              0.1027,
              0.0516,
              0.0
            ],
            [
              1.4479,
              0.0393,
              0.0069,
              0.0
            ],
            [
              1.7843,
              0.1105,
              0.0539,
              0.0129
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "mid": {
        "available": true,
        "cost": {
          "MAE": 38.05,
          "RMSE": 60.293,
          "R2": -0.1914,
          "MAPE": 186.02,
          "rows": 1208,
          "unique_projects": 389
        },
        "delay": {
          "MAE": 399.485,
          "RMSE": 596.819,
          "R2": -0.0619,
          "MAPE": 103.725,
          "rows": 1208,
          "unique_projects": 389
        },
        "risk": {
          "accuracy": 0.337,
          "macro_precision": 0.4237,
          "macro_recall": 0.2744,
          "macro_f1": 0.1764,
          "confusion_matrix": [
            [
              8.5441,
              0.0,
              0.1125,
              0.0319
            ],
            [
              5.5091,
              0.2896,
              0.1278,
              0.0
            ],
            [
              5.2508,
              0.2048,
              0.1972,
              0.0
            ],
            [
              5.8716,
              0.7115,
              0.3759,
              0.2191
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "late": {
        "available": true,
        "cost": {
          "MAE": 35.124,
          "RMSE": 52.405,
          "R2": 0.0118,
          "MAPE": 214.343,
          "rows": 1741,
          "unique_projects": 512
        },
        "delay": {
          "MAE": 447.44,
          "RMSE": 684.177,
          "R2": -0.1015,
          "MAPE": 113.239,
          "rows": 1741,
          "unique_projects": 512
        },
        "risk": {
          "accuracy": 0.2455,
          "macro_precision": 0.2865,
          "macro_recall": 0.283,
          "macro_f1": 0.1714,
          "confusion_matrix": [
            [
              8.9436,
              0.25,
              0.0731,
              0.1839
            ],
            [
              10.6329,
              0.759,
              0.3163,
              0.1683
            ],
            [
              7.6093,
              1.6241,
              0.6525,
              0.4161
            ],
            [
              9.1849,
              1.9703,
              1.8828,
              0.8064
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "very_late": {
        "available": true,
        "cost": {
          "MAE": 39.683,
          "RMSE": 72.569,
          "R2": 0.279,
          "MAPE": 197.783,
          "rows": 8615,
          "unique_projects": 736
        },
        "delay": {
          "MAE": 491.159,
          "RMSE": 749.016,
          "R2": 0.6229,
          "MAPE": 43.515,
          "rows": 8615,
          "unique_projects": 736
        },
        "risk": {
          "accuracy": 0.6425,
          "macro_precision": 0.4259,
          "macro_recall": 0.5564,
          "macro_f1": 0.4504,
          "confusion_matrix": [
            [
              2.4039,
              1.1554,
              0.0941,
              0.073
            ],
            [
              4.3768,
              9.213,
              6.2141,
              3.0164
            ],
            [
              2.9891,
              5.859,
              13.4354,
              6.8479
            ],
            [
              3.2613,
              15.4763,
              22.9032,
              104.8405
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      }
    }
  }
}
```

### SHAP / importance

```json
{
  "cost": {
    "method": "mean_absolute_shap",
    "features": [
      {
        "feature": "expenditure_ratio",
        "importance": 0.335156
      },
      {
        "feature": "cost_escalation_percentage",
        "importance": 0.195355
      },
      {
        "feature": "planned_duration_days",
        "importance": 0.072543
      },
      {
        "feature": "cumulative_expenditure_cr",
        "importance": 0.060439
      },
      {
        "feature": "agency_delay_rate",
        "importance": 0.058967
      },
      {
        "feature": "revised_cost_cr",
        "importance": 0.043675
      },
      {
        "feature": "expected_progress_percentage",
        "importance": 0.032704
      },
      {
        "feature": "approved_cost_cr",
        "importance": 0.027491
      },
      {
        "feature": "schedule_slippage_ratio",
        "importance": 0.026444
      },
      {
        "feature": "sector",
        "importance": 0.026433
      },
      {
        "feature": "duration_ratio",
        "importance": 0.023147
      },
      {
        "feature": "sector_average_delay",
        "importance": 0.019171
      },
      {
        "feature": "implementing_agency",
        "importance": 0.01708
      },
      {
        "feature": "elapsed_duration_days",
        "importance": 0.014677
      },
      {
        "feature": "agency_average_delay",
        "importance": 0.013403
      },
      {
        "feature": "schedule_slippage_days",
        "importance": 0.010522
      },
      {
        "feature": "missingindicator_schedule_slippage_days",
        "importance": 0.007006
      },
      {
        "feature": "agency_cost_overrun_rate",
        "importance": 0.005551
      },
      {
        "feature": "agency_average_cost_overrun",
        "importance": 0.003262
      },
      {
        "feature": "sector_average_cost_overrun",
        "importance": 0.003219
      },
      {
        "feature": "project_size_category",
        "importance": 0.000911
      },
      {
        "feature": "missingindicator_schedule_slippage_ratio",
        "importance": 0.000904
      },
      {
        "feature": "missingindicator_revised_cost_cr",
        "importance": 0.000558
      },
      {
        "feature": "missingindicator_planned_duration_days",
        "importance": 0.000489
      },
      {
        "feature": "missingindicator_cost_growth_velocity_6m",
        "importance": 0.000298
      },
      {
        "feature": "missingindicator_cost_escalation_percentage",
        "importance": 0.000191
      },
      {
        "feature": "cost_growth_velocity_6m",
        "importance": 0.000187
      },
      {
        "feature": "missingindicator_sector_average_delay",
        "importance": 0.000144
      },
      {
        "feature": "missingindicator_duration_ratio",
        "importance": 7.4e-05
      },
      {
        "feature": "cost_growth_velocity_3m",
        "importance": 0.0
      },
      {
        "feature": "cost_acceleration",
        "importance": 0.0
      },
      {
        "feature": "sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_average_cost_overrun",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_expenditure_ratio",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_expected_progress_percentage",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_growth_velocity_3m",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_acceleration",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_average_delay",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_average_cost_overrun",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_cost_overrun_rate",
        "importance": 0.0
      }
    ]
  },
  "delay": {
    "method": "tree_feature_importance",
    "features": [
      {
        "feature": "elapsed_duration_days",
        "importance": 0.122344
      },
      {
        "feature": "schedule_slippage_days",
        "importance": 0.10803
      },
      {
        "feature": "duration_ratio",
        "importance": 0.101991
      },
      {
        "feature": "approved_cost_cr",
        "importance": 0.081414
      },
      {
        "feature": "planned_duration_days",
        "importance": 0.075375
      },
      {
        "feature": "revised_cost_cr",
        "importance": 0.072467
      },
      {
        "feature": "expenditure_ratio",
        "importance": 0.069783
      },
      {
        "feature": "sector",
        "importance": 0.069336
      },
      {
        "feature": "cost_escalation_percentage",
        "importance": 0.061284
      },
      {
        "feature": "implementing_agency",
        "importance": 0.053903
      },
      {
        "feature": "cumulative_expenditure_cr",
        "importance": 0.051666
      },
      {
        "feature": "sector_average_delay",
        "importance": 0.031313
      },
      {
        "feature": "schedule_slippage_ratio",
        "importance": 0.027287
      },
      {
        "feature": "agency_average_delay",
        "importance": 0.013867
      },
      {
        "feature": "sector_average_cost_overrun",
        "importance": 0.011631
      },
      {
        "feature": "agency_delay_rate",
        "importance": 0.011407
      },
      {
        "feature": "agency_average_cost_overrun",
        "importance": 0.008723
      },
      {
        "feature": "missingindicator_revised_cost_cr",
        "importance": 0.007828
      },
      {
        "feature": "agency_cost_overrun_rate",
        "importance": 0.005144
      },
      {
        "feature": "expected_progress_percentage",
        "importance": 0.004921
      },
      {
        "feature": "project_size_category",
        "importance": 0.003355
      },
      {
        "feature": "missingindicator_schedule_slippage_ratio",
        "importance": 0.003131
      },
      {
        "feature": "missingindicator_schedule_slippage_days",
        "importance": 0.001789
      },
      {
        "feature": "missingindicator_cost_growth_velocity_3m",
        "importance": 0.000895
      },
      {
        "feature": "cost_growth_velocity_6m",
        "importance": 0.000447
      },
      {
        "feature": "missingindicator_cost_growth_velocity_6m",
        "importance": 0.000447
      },
      {
        "feature": "missingindicator_cost_escalation_percentage",
        "importance": 0.000224
      },
      {
        "feature": "cost_growth_velocity_3m",
        "importance": 0.0
      },
      {
        "feature": "cost_acceleration",
        "importance": 0.0
      },
      {
        "feature": "sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_average_delay",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_average_cost_overrun",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_expenditure_ratio",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_planned_duration_days",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_duration_ratio",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_expected_progress_percentage",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_acceleration",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_average_delay",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_average_cost_overrun",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_agency_cost_overrun_rate",
        "importance": 0.0
      }
    ]
  },
  "risk": {
    "method": "tree_feature_importance",
    "features": [
      {
        "feature": "sector",
        "importance": 0.119786
      },
      {
        "feature": "schedule_slippage_days",
        "importance": 0.117139
      },
      {
        "feature": "duration_ratio",
        "importance": 0.112622
      },
      {
        "feature": "schedule_slippage_ratio",
        "importance": 0.102202
      },
      {
        "feature": "expected_progress_percentage",
        "importance": 0.081283
      },
      {
        "feature": "elapsed_duration_days",
        "importance": 0.073392
      },
      {
        "feature": "implementing_agency",
        "importance": 0.054434
      },
      {
        "feature": "expenditure_ratio",
        "importance": 0.040274
      },
      {
        "feature": "planned_duration_days",
        "importance": 0.03673
      },
      {
        "feature": "cumulative_expenditure_cr",
        "importance": 0.025535
      },
      {
        "feature": "approved_cost_cr",
        "importance": 0.025117
      },
      {
        "feature": "revised_cost_cr",
        "importance": 0.02359
      },
      {
        "feature": "cost_escalation_percentage",
        "importance": 0.022379
      },
      {
        "feature": "agency_delay_rate",
        "importance": 0.018505
      },
      {
        "feature": "sector_average_cost_overrun",
        "importance": 0.017575
      },
      {
        "feature": "agency_average_cost_overrun",
        "importance": 0.01568
      },
      {
        "feature": "agency_average_delay",
        "importance": 0.015091
      },
      {
        "feature": "sector_average_delay",
        "importance": 0.014844
      },
      {
        "feature": "agency_cost_overrun_rate",
        "importance": 0.012023
      },
      {
        "feature": "project_size_category",
        "importance": 0.011698
      },
      {
        "feature": "missingindicator_revised_cost_cr",
        "importance": 0.008683
      },
      {
        "feature": "missingindicator_cost_escalation_percentage",
        "importance": 0.007772
      },
      {
        "feature": "missingindicator_cost_growth_velocity_6m",
        "importance": 0.005125
      },
      {
        "feature": "missingindicator_cost_acceleration",
        "importance": 0.004951
      },
      {
        "feature": "missingindicator_cost_growth_velocity_3m",
        "importance": 0.004599
      },
      {
        "feature": "cost_growth_velocity_6m",
        "importance": 0.004566
      },
      {
        "feature": "cost_acceleration",
        "importance": 0.002801
      },
      {
        "feature": "cost_growth_velocity_3m",
        "importance": 0.002593
      },
      {
        "feature": "missingindicator_agency_average_delay",
        "importance": 0.001937
      },
      {
        "feature": "missingindicator_schedule_slippage_ratio",
        "importance": 0.001757
      },
      {
        "feature": "missingindicator_duration_ratio",
        "importance": 0.001609
      },
      {
        "feature": "missingindicator_agency_average_cost_overrun",
        "importance": 0.001586
      },
      {
        "feature": "missingindicator_expected_progress_percentage",
        "importance": 0.001522
      },
      {
        "feature": "missingindicator_planned_duration_days",
        "importance": 0.00147
      },
      {
        "feature": "missingindicator_agency_cost_overrun_rate",
        "importance": 0.001398
      },
      {
        "feature": "missingindicator_agency_delay_rate",
        "importance": 0.001394
      },
      {
        "feature": "missingindicator_sector_cost_overrun_rate",
        "importance": 0.001382
      },
      {
        "feature": "missingindicator_schedule_slippage_days",
        "importance": 0.001334
      },
      {
        "feature": "missingindicator_sector_average_delay",
        "importance": 0.001192
      },
      {
        "feature": "missingindicator_sector_average_cost_overrun",
        "importance": 0.001157
      },
      {
        "feature": "missingindicator_sector_delay_rate",
        "importance": 0.001131
      },
      {
        "feature": "missingindicator_expenditure_ratio",
        "importance": 0.000139
      },
      {
        "feature": "sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "sector_cost_overrun_rate",
        "importance": 0.0
      }
    ]
  }
}
```

Conclusion for this window: both cost and delay MAE improved.
Each ablation repeats the same training-only candidate selection protocol; the final holdout remains untouched. Trajectory and agency-prior effects are interpreted metric by metric and do not inherit the full model's overall improvement claim.

## Window 2015_2021

### Baseline versus lifecycle

Training: 13736 snapshots / 892 projects. Test: 10967 snapshots / 691 projects.

Selected regressors: cost=xgboost; delay=xgboost. Risk uses the documented Random Forest classifier.

| Metric | Five-feature baseline | Monthly lifecycle |
|---|---:|---:|
| Cost MAE | 36.249 | 28.283 |
| Cost RMSE | 72.704 | 60.219 |
| Cost R2 | 0.0612 | 0.356 |
| Delay MAE | 759.933 | 530.599 |
| Delay RMSE | 1052.501 | 717.985 |
| Delay R2 | -0.027 | 0.5221 |
| Risk accuracy | 0.3963 | 0.5855 |
| Risk macro F1 | 0.2526 | 0.4278 |

Primary MAE improvement: cost 22.0%; delay 30.2%.

### Feature audit

Retained (25): approved_cost_cr, cumulative_expenditure_cr, expenditure_ratio, schedule_slippage_days, schedule_slippage_ratio, elapsed_duration_days, planned_duration_days, duration_ratio, expected_progress_percentage, revised_cost_cr, cost_escalation_percentage, sector, project_size_category, implementing_agency, cost_growth_velocity_3m, cost_growth_velocity_6m, cost_acceleration, sector_average_delay, sector_average_cost_overrun, sector_delay_rate, sector_cost_overrun_rate, agency_average_delay, agency_average_cost_overrun, agency_delay_rate, agency_cost_overrun_rate.

Rejected (7): physical_progress (availability below 10.0%); progress_deviation (availability below 10.0%); current_schedule_status (availability below 10.0%); ministry (availability below 10.0%); progress_velocity_3m (availability below 10.0%); progress_velocity_6m (availability below 10.0%); progress_acceleration (availability below 10.0%).

| Feature | Available | Missing | Years | Projects | As-of safe | Decision | Reason |
|---|---:|---:|---:|---:|---|---|---|
| approved_cost_cr | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| cumulative_expenditure_cr | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| expenditure_ratio | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| physical_progress | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| schedule_slippage_days | 95.76% | 4.24% | 18 | 883 | yes | keep | observed and variable in the training window |
| schedule_slippage_ratio | 95.19% | 4.81% | 18 | 875 | yes | keep | observed and variable in the training window |
| elapsed_duration_days | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| planned_duration_days | 95.76% | 4.24% | 18 | 883 | yes | keep | observed and variable in the training window |
| duration_ratio | 95.19% | 4.81% | 18 | 875 | yes | keep | observed and variable in the training window |
| expected_progress_percentage | 95.19% | 4.81% | 18 | 875 | yes | keep | observed and variable in the training window |
| progress_deviation | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| revised_cost_cr | 99.99% | 0.01% | 18 | 892 | yes | keep | observed and variable in the training window |
| cost_escalation_percentage | 99.99% | 0.01% | 18 | 892 | yes | keep | observed and variable in the training window |
| current_schedule_status | 0.2% | 99.8% | 6 | 4 | yes | remove | availability below 10.0% |
| sector | 97.83% | 2.17% | 18 | 892 | yes | keep | observed and variable in the training window |
| project_size_category | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| implementing_agency | 95.61% | 4.39% | 17 | 891 | yes | keep | observed and variable in the training window |
| ministry | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| cost_growth_velocity_3m | 93.22% | 6.78% | 18 | 878 | yes | keep | observed and variable in the training window |
| cost_growth_velocity_6m | 94.21% | 5.79% | 18 | 879 | yes | keep | observed and variable in the training window |
| cost_acceleration | 93.22% | 6.78% | 18 | 878 | yes | keep | observed and variable in the training window |
| progress_velocity_3m | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| progress_velocity_6m | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| progress_acceleration | 0.0% | 100.0% | 0 | 0 | yes | remove | availability below 10.0% |
| sector_average_delay | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| sector_average_cost_overrun | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| sector_delay_rate | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| sector_cost_overrun_rate | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| agency_average_delay | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| agency_average_cost_overrun | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| agency_delay_rate | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |
| agency_cost_overrun_rate | 100.0% | 0.0% | 18 | 892 | yes | keep | observed and variable in the training window |

### Lifecycle-stage evaluation

```json
{
  "early": {
    "available": true,
    "cost": {
      "MAE": 37.31,
      "RMSE": 50.529,
      "R2": -1.1718,
      "MAPE": 261.756,
      "rows": 169,
      "unique_projects": 93
    },
    "delay": {
      "MAE": 616.242,
      "RMSE": 821.79,
      "R2": 0.0469,
      "MAPE": 79.673,
      "rows": 169,
      "unique_projects": 93
    },
    "risk": {
      "accuracy": 0.1539,
      "macro_precision": 0.2269,
      "macro_recall": 0.2181,
      "macro_f1": 0.1444,
      "confusion_matrix": [
        [
          0.1723,
          0.0345,
          0.0909,
          0.0909
        ],
        [
          0.339,
          0.1894,
          0.0556,
          0.0
        ],
        [
          0.5243,
          0.382,
          0.0,
          0.0
        ],
        [
          0.5575,
          0.4983,
          0.2585,
          0.1535
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  },
  "mid": {
    "available": true,
    "cost": {
      "MAE": 36.712,
      "RMSE": 50.871,
      "R2": -0.218,
      "MAPE": 234.077,
      "rows": 655,
      "unique_projects": 239
    },
    "delay": {
      "MAE": 488.195,
      "RMSE": 725.502,
      "R2": 0.1458,
      "MAPE": 106.886,
      "rows": 655,
      "unique_projects": 239
    },
    "risk": {
      "accuracy": 0.3676,
      "macro_precision": 0.3862,
      "macro_recall": 0.2931,
      "macro_f1": 0.2426,
      "confusion_matrix": [
        [
          5.7467,
          0.4824,
          1.8594,
          0.0345
        ],
        [
          0.7079,
          0.4601,
          0.1181,
          0.0625
        ],
        [
          1.5857,
          1.4302,
          0.1297,
          0.0
        ],
        [
          1.9634,
          2.6766,
          0.8224,
          0.4906
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  },
  "late": {
    "available": true,
    "cost": {
      "MAE": 35.602,
      "RMSE": 89.546,
      "R2": 0.0642,
      "MAPE": 198.488,
      "rows": 1126,
      "unique_projects": 368
    },
    "delay": {
      "MAE": 580.73,
      "RMSE": 780.064,
      "R2": -0.0051,
      "MAPE": 95.328,
      "rows": 1126,
      "unique_projects": 368
    },
    "risk": {
      "accuracy": 0.2311,
      "macro_precision": 0.3058,
      "macro_recall": 0.2734,
      "macro_f1": 0.2349,
      "confusion_matrix": [
        [
          1.7527,
          0.507,
          2.2632,
          0.6607
        ],
        [
          1.2813,
          1.6693,
          1.1737,
          0.2198
        ],
        [
          1.2728,
          2.4263,
          1.1398,
          0.2521
        ],
        [
          1.7334,
          5.7912,
          4.5092,
          2.0769
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  },
  "very_late": {
    "available": true,
    "cost": {
      "MAE": 24.591,
      "RMSE": 54.028,
      "R2": 0.3799,
      "MAPE": 131.743,
      "rows": 8159,
      "unique_projects": 652
    },
    "delay": {
      "MAE": 518.037,
      "RMSE": 696.434,
      "R2": 0.5668,
      "MAPE": 48.459,
      "rows": 8159,
      "unique_projects": 652
    },
    "risk": {
      "accuracy": 0.6722,
      "macro_precision": 0.4669,
      "macro_recall": 0.3811,
      "macro_f1": 0.4009,
      "confusion_matrix": [
        [
          0.579,
          1.2333,
          1.7581,
          0.9794
        ],
        [
          0.2166,
          6.9592,
          3.6316,
          10.1045
        ],
        [
          0.25,
          3.8268,
          7.5816,
          18.9374
        ],
        [
          0.2655,
          5.0353,
          23.3991,
          127.6825
        ]
      ],
      "labels": [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
      ]
    }
  }
}
```

### Ablations

```json
{
  "without_revised_cost": {
    "features": [
      "approved_cost_cr",
      "sector_average_delay",
      "sector_average_cost_overrun",
      "sector",
      "project_size_category",
      "cumulative_expenditure_cr",
      "expenditure_ratio",
      "schedule_slippage_days",
      "schedule_slippage_ratio",
      "elapsed_duration_days",
      "planned_duration_days",
      "duration_ratio",
      "expected_progress_percentage",
      "implementing_agency",
      "cost_growth_velocity_3m",
      "cost_growth_velocity_6m",
      "cost_acceleration",
      "sector_delay_rate",
      "sector_cost_overrun_rate",
      "agency_average_delay",
      "agency_average_cost_overrun",
      "agency_delay_rate",
      "agency_cost_overrun_rate"
    ],
    "selected_algorithms": {
      "cost": "lightgbm",
      "delay": "extra_trees"
    },
    "internal_algorithm_comparisons": {
      "cost": [
        {
          "algorithm": "lightgbm",
          "MAE": 33.723,
          "RMSE": 71.5,
          "R2": 0.4493,
          "MAPE": 189.631,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "xgboost",
          "MAE": 34.984,
          "RMSE": 72.855,
          "R2": 0.4282,
          "MAPE": 182.767,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "extra_trees",
          "MAE": 35.171,
          "RMSE": 74.721,
          "R2": 0.3986,
          "MAPE": 166.446,
          "rows": 2613,
          "unique_projects": 154
        }
      ],
      "delay": [
        {
          "algorithm": "lightgbm",
          "MAE": 478.663,
          "RMSE": 673.198,
          "R2": 0.6608,
          "MAPE": 49.661,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "xgboost",
          "MAE": 477.192,
          "RMSE": 678.677,
          "R2": 0.6552,
          "MAPE": 47.402,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "extra_trees",
          "MAE": 470.81,
          "RMSE": 677.533,
          "R2": 0.6564,
          "MAPE": 55.936,
          "rows": 2613,
          "unique_projects": 154
        }
      ]
    },
    "metrics": {
      "cost": {
        "MAE": 29.769,
        "RMSE": 62.767,
        "R2": 0.3003,
        "MAPE": 166.396,
        "rows": 10967,
        "unique_projects": 691
      },
      "delay": {
        "MAE": 549.031,
        "RMSE": 726.483,
        "R2": 0.5107,
        "MAPE": 74.522,
        "rows": 10967,
        "unique_projects": 691
      },
      "risk": {
        "accuracy": 0.5734,
        "macro_precision": 0.4146,
        "macro_recall": 0.4326,
        "macro_f1": 0.4192,
        "confusion_matrix": [
          [
            9.0464,
            1.8205,
            6.1723,
            1.3867
          ],
          [
            2.7119,
            9.6735,
            10.1362,
            7.9776
          ],
          [
            3.678,
            11.4303,
            9.6709,
            19.5334
          ],
          [
            4.8773,
            21.5647,
            30.2572,
            134.9963
          ]
        ],
        "labels": [
          "LOW",
          "MEDIUM",
          "HIGH",
          "CRITICAL"
        ]
      }
    },
    "lifecycle_stages": {
      "early": {
        "available": true,
        "cost": {
          "MAE": 34.028,
          "RMSE": 47.438,
          "R2": -0.9142,
          "MAPE": 266.202,
          "rows": 169,
          "unique_projects": 93
        },
        "delay": {
          "MAE": 568.398,
          "RMSE": 834.158,
          "R2": 0.018,
          "MAPE": 67.933,
          "rows": 169,
          "unique_projects": 93
        },
        "risk": {
          "accuracy": 0.1425,
          "macro_precision": 0.2078,
          "macro_recall": 0.2216,
          "macro_f1": 0.1296,
          "confusion_matrix": [
            [
              0.2328,
              0.0649,
              0.0,
              0.0909
            ],
            [
              0.3586,
              0.1171,
              0.1082,
              0.0
            ],
            [
              0.5297,
              0.3767,
              0.0,
              0.0
            ],
            [
              0.5382,
              0.5247,
              0.2779,
              0.127
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "mid": {
        "available": true,
        "cost": {
          "MAE": 31.449,
          "RMSE": 48.709,
          "R2": -0.1166,
          "MAPE": 232.496,
          "rows": 655,
          "unique_projects": 239
        },
        "delay": {
          "MAE": 574.663,
          "RMSE": 746.056,
          "R2": 0.0967,
          "MAPE": 133.627,
          "rows": 655,
          "unique_projects": 239
        },
        "risk": {
          "accuracy": 0.3632,
          "macro_precision": 0.4208,
          "macro_recall": 0.2774,
          "macro_f1": 0.233,
          "confusion_matrix": [
            [
              5.8181,
              0.1034,
              2.2014,
              0.0
            ],
            [
              0.7284,
              0.3871,
              0.2332,
              0.0
            ],
            [
              1.6417,
              1.3986,
              0.1053,
              0.0
            ],
            [
              2.0861,
              2.6619,
              0.7711,
              0.4339
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "late": {
        "available": true,
        "cost": {
          "MAE": 37.314,
          "RMSE": 92.812,
          "R2": -0.0053,
          "MAPE": 220.865,
          "rows": 1126,
          "unique_projects": 368
        },
        "delay": {
          "MAE": 553.133,
          "RMSE": 739.761,
          "R2": 0.0961,
          "MAPE": 129.097,
          "rows": 1126,
          "unique_projects": 368
        },
        "risk": {
          "accuracy": 0.2349,
          "macro_precision": 0.3537,
          "macro_recall": 0.3016,
          "macro_f1": 0.2427,
          "confusion_matrix": [
            [
              2.1695,
              0.3338,
              2.3412,
              0.3391
            ],
            [
              1.2656,
              1.8566,
              1.1505,
              0.0714
            ],
            [
              1.2228,
              2.5343,
              1.3339,
              0.0
            ],
            [
              1.871,
              6.3128,
              4.5386,
              1.3882
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "very_late": {
        "available": true,
        "cost": {
          "MAE": 26.807,
          "RMSE": 56.946,
          "R2": 0.3111,
          "MAPE": 153.353,
          "rows": 8159,
          "unique_projects": 652
        },
        "delay": {
          "MAE": 544.446,
          "RMSE": 712.622,
          "R2": 0.5465,
          "MAPE": 63.486,
          "rows": 8159,
          "unique_projects": 652
        },
        "risk": {
          "accuracy": 0.654,
          "macro_precision": 0.435,
          "macro_recall": 0.3591,
          "macro_f1": 0.379,
          "confusion_matrix": [
            [
              0.8259,
              1.2333,
              1.5552,
              0.9354
            ],
            [
              0.3594,
              4.1744,
              8.6443,
              7.7337
            ],
            [
              0.2632,
              3.6507,
              7.5662,
              19.1158
            ],
            [
              0.3325,
              6.233,
              23.4452,
              126.3718
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      }
    }
  },
  "snapshot_only": {
    "features": [
      "approved_cost_cr",
      "sector_average_delay",
      "sector_average_cost_overrun",
      "sector",
      "project_size_category",
      "cumulative_expenditure_cr",
      "expenditure_ratio",
      "schedule_slippage_days",
      "schedule_slippage_ratio",
      "elapsed_duration_days",
      "planned_duration_days",
      "duration_ratio",
      "expected_progress_percentage",
      "revised_cost_cr",
      "cost_escalation_percentage",
      "implementing_agency",
      "sector_delay_rate",
      "sector_cost_overrun_rate",
      "agency_average_delay",
      "agency_average_cost_overrun",
      "agency_delay_rate",
      "agency_cost_overrun_rate"
    ],
    "selected_algorithms": {
      "cost": "xgboost",
      "delay": "extra_trees"
    },
    "internal_algorithm_comparisons": {
      "cost": [
        {
          "algorithm": "lightgbm",
          "MAE": 33.858,
          "RMSE": 69.683,
          "R2": 0.4769,
          "MAPE": 186.74,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "xgboost",
          "MAE": 32.736,
          "RMSE": 68.718,
          "R2": 0.4913,
          "MAPE": 167.444,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "extra_trees",
          "MAE": 34.321,
          "RMSE": 71.509,
          "R2": 0.4492,
          "MAPE": 167.737,
          "rows": 2613,
          "unique_projects": 154
        }
      ],
      "delay": [
        {
          "algorithm": "lightgbm",
          "MAE": 474.172,
          "RMSE": 667.83,
          "R2": 0.6662,
          "MAPE": 47.808,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "xgboost",
          "MAE": 476.022,
          "RMSE": 674.231,
          "R2": 0.6597,
          "MAPE": 47.063,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "extra_trees",
          "MAE": 467.348,
          "RMSE": 676.069,
          "R2": 0.6579,
          "MAPE": 53.402,
          "rows": 2613,
          "unique_projects": 154
        }
      ]
    },
    "metrics": {
      "cost": {
        "MAE": 28.237,
        "RMSE": 60.559,
        "R2": 0.3487,
        "MAPE": 143.456,
        "rows": 10967,
        "unique_projects": 691
      },
      "delay": {
        "MAE": 537.991,
        "RMSE": 718.477,
        "R2": 0.5214,
        "MAPE": 72.64,
        "rows": 10967,
        "unique_projects": 691
      },
      "risk": {
        "accuracy": 0.5947,
        "macro_precision": 0.4286,
        "macro_recall": 0.4508,
        "macro_f1": 0.4335,
        "confusion_matrix": [
          [
            8.4112,
            3.412,
            4.7854,
            1.8173
          ],
          [
            2.3734,
            13.3645,
            5.8072,
            8.9542
          ],
          [
            3.1991,
            12.739,
            7.9666,
            20.408
          ],
          [
            4.4512,
            21.6107,
            25.9321,
            139.7016
          ]
        ],
        "labels": [
          "LOW",
          "MEDIUM",
          "HIGH",
          "CRITICAL"
        ]
      }
    },
    "lifecycle_stages": {
      "early": {
        "available": true,
        "cost": {
          "MAE": 38.088,
          "RMSE": 50.963,
          "R2": -1.2093,
          "MAPE": 271.216,
          "rows": 169,
          "unique_projects": 93
        },
        "delay": {
          "MAE": 574.131,
          "RMSE": 839.879,
          "R2": 0.0045,
          "MAPE": 73.846,
          "rows": 169,
          "unique_projects": 93
        },
        "risk": {
          "accuracy": 0.1318,
          "macro_precision": 0.1617,
          "macro_recall": 0.1829,
          "macro_f1": 0.121,
          "confusion_matrix": [
            [
              0.1246,
              0.0821,
              0.0,
              0.1818
            ],
            [
              0.339,
              0.1894,
              0.0556,
              0.0
            ],
            [
              0.4819,
              0.4245,
              0.0,
              0.0
            ],
            [
              0.4778,
              0.5834,
              0.2796,
              0.127
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "mid": {
        "available": true,
        "cost": {
          "MAE": 36.311,
          "RMSE": 50.635,
          "R2": -0.2067,
          "MAPE": 230.666,
          "rows": 655,
          "unique_projects": 239
        },
        "delay": {
          "MAE": 569.534,
          "RMSE": 742.357,
          "R2": 0.1057,
          "MAPE": 137.642,
          "rows": 655,
          "unique_projects": 239
        },
        "risk": {
          "accuracy": 0.355,
          "macro_precision": 0.3717,
          "macro_recall": 0.2916,
          "macro_f1": 0.2377,
          "confusion_matrix": [
            [
              5.5124,
              0.5486,
              1.9906,
              0.0714
            ],
            [
              0.6579,
              0.5101,
              0.1181,
              0.0625
            ],
            [
              1.4382,
              1.6158,
              0.0917,
              0.0
            ],
            [
              1.8483,
              2.8666,
              0.7607,
              0.4775
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "late": {
        "available": true,
        "cost": {
          "MAE": 35.922,
          "RMSE": 90.566,
          "R2": 0.0428,
          "MAPE": 201.25,
          "rows": 1126,
          "unique_projects": 368
        },
        "delay": {
          "MAE": 550.906,
          "RMSE": 739.794,
          "R2": 0.096,
          "MAPE": 133.653,
          "rows": 1126,
          "unique_projects": 368
        },
        "risk": {
          "accuracy": 0.2692,
          "macro_precision": 0.3447,
          "macro_recall": 0.3177,
          "macro_f1": 0.2651,
          "confusion_matrix": [
            [
              1.6438,
              1.2239,
              1.7951,
              0.5208
            ],
            [
              1.1511,
              2.4479,
              0.5968,
              0.1484
            ],
            [
              1.1031,
              2.7251,
              1.0518,
              0.211
            ],
            [
              1.8214,
              6.0612,
              3.6381,
              2.59
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "very_late": {
        "available": true,
        "cost": {
          "MAE": 24.509,
          "RMSE": 54.352,
          "R2": 0.3724,
          "MAPE": 127.778,
          "rows": 8159,
          "unique_projects": 652
        },
        "delay": {
          "MAE": 530.258,
          "RMSE": 703.302,
          "R2": 0.5583,
          "MAPE": 60.124,
          "rows": 8159,
          "unique_projects": 652
        },
        "risk": {
          "accuracy": 0.6782,
          "macro_precision": 0.501,
          "macro_recall": 0.4057,
          "macro_f1": 0.4327,
          "confusion_matrix": [
            [
              1.1304,
              1.5255,
              0.9145,
              0.9794
            ],
            [
              0.2254,
              7.1069,
              5.0368,
              8.5429
            ],
            [
              0.176,
              4.5316,
              6.3141,
              19.5742
            ],
            [
              0.2692,
              6.2387,
              20.3405,
              129.534
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      }
    }
  },
  "without_agency_priors": {
    "features": [
      "approved_cost_cr",
      "sector_average_delay",
      "sector_average_cost_overrun",
      "sector",
      "project_size_category",
      "cumulative_expenditure_cr",
      "expenditure_ratio",
      "schedule_slippage_days",
      "schedule_slippage_ratio",
      "elapsed_duration_days",
      "planned_duration_days",
      "duration_ratio",
      "expected_progress_percentage",
      "revised_cost_cr",
      "cost_escalation_percentage",
      "implementing_agency",
      "cost_growth_velocity_3m",
      "cost_growth_velocity_6m",
      "cost_acceleration",
      "sector_delay_rate",
      "sector_cost_overrun_rate"
    ],
    "selected_algorithms": {
      "cost": "xgboost",
      "delay": "extra_trees"
    },
    "internal_algorithm_comparisons": {
      "cost": [
        {
          "algorithm": "lightgbm",
          "MAE": 33.97,
          "RMSE": 70.738,
          "R2": 0.461,
          "MAPE": 190.266,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "xgboost",
          "MAE": 32.839,
          "RMSE": 67.981,
          "R2": 0.5022,
          "MAPE": 175.708,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "extra_trees",
          "MAE": 33.763,
          "RMSE": 70.529,
          "R2": 0.4642,
          "MAPE": 159.507,
          "rows": 2613,
          "unique_projects": 154
        }
      ],
      "delay": [
        {
          "algorithm": "lightgbm",
          "MAE": 486.389,
          "RMSE": 676.281,
          "R2": 0.6577,
          "MAPE": 46.248,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "xgboost",
          "MAE": 484.812,
          "RMSE": 680.188,
          "R2": 0.6537,
          "MAPE": 47.31,
          "rows": 2613,
          "unique_projects": 154
        },
        {
          "algorithm": "extra_trees",
          "MAE": 473.443,
          "RMSE": 682.652,
          "R2": 0.6512,
          "MAPE": 53.028,
          "rows": 2613,
          "unique_projects": 154
        }
      ]
    },
    "metrics": {
      "cost": {
        "MAE": 29.035,
        "RMSE": 61.066,
        "R2": 0.3377,
        "MAPE": 161.057,
        "rows": 10967,
        "unique_projects": 691
      },
      "delay": {
        "MAE": 533.505,
        "RMSE": 718.487,
        "R2": 0.5214,
        "MAPE": 70.27,
        "rows": 10967,
        "unique_projects": 691
      },
      "risk": {
        "accuracy": 0.5741,
        "macro_precision": 0.4152,
        "macro_recall": 0.4572,
        "macro_f1": 0.4278,
        "confusion_matrix": [
          [
            9.9433,
            2.7197,
            4.4104,
            1.3525
          ],
          [
            3.1236,
            12.7872,
            6.1817,
            8.4067
          ],
          [
            3.6563,
            12.9558,
            7.7875,
            19.913
          ],
          [
            6.3262,
            23.2051,
            29.0901,
            133.0742
          ]
        ],
        "labels": [
          "LOW",
          "MEDIUM",
          "HIGH",
          "CRITICAL"
        ]
      }
    },
    "lifecycle_stages": {
      "early": {
        "available": true,
        "cost": {
          "MAE": 37.616,
          "RMSE": 50.074,
          "R2": -1.1328,
          "MAPE": 277.242,
          "rows": 169,
          "unique_projects": 93
        },
        "delay": {
          "MAE": 570.448,
          "RMSE": 837.444,
          "R2": 0.0103,
          "MAPE": 69.938,
          "rows": 169,
          "unique_projects": 93
        },
        "risk": {
          "accuracy": 0.1122,
          "macro_precision": 0.1108,
          "macro_recall": 0.1799,
          "macro_f1": 0.1022,
          "confusion_matrix": [
            [
              0.1419,
              0.0649,
              0.0,
              0.1818
            ],
            [
              0.339,
              0.1894,
              0.0556,
              0.0
            ],
            [
              0.4992,
              0.4071,
              0.0,
              0.0
            ],
            [
              0.5991,
              0.5388,
              0.2857,
              0.0442
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "mid": {
        "available": true,
        "cost": {
          "MAE": 36.616,
          "RMSE": 51.024,
          "R2": -0.2253,
          "MAPE": 223.755,
          "rows": 655,
          "unique_projects": 239
        },
        "delay": {
          "MAE": 528.412,
          "RMSE": 718.883,
          "R2": 0.1613,
          "MAPE": 123.434,
          "rows": 655,
          "unique_projects": 239
        },
        "risk": {
          "accuracy": 0.3791,
          "macro_precision": 0.3637,
          "macro_recall": 0.2898,
          "macro_f1": 0.2241,
          "confusion_matrix": [
            [
              6.3013,
              0.3481,
              1.4021,
              0.0714
            ],
            [
              0.7287,
              0.4393,
              0.1806,
              0.0
            ],
            [
              1.6688,
              1.4269,
              0.05,
              0.0
            ],
            [
              2.5081,
              2.4044,
              0.7919,
              0.2486
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "late": {
        "available": true,
        "cost": {
          "MAE": 36.518,
          "RMSE": 90.967,
          "R2": 0.0343,
          "MAPE": 217.626,
          "rows": 1126,
          "unique_projects": 368
        },
        "delay": {
          "MAE": 550.544,
          "RMSE": 740.269,
          "R2": 0.0949,
          "MAPE": 123.865,
          "rows": 1126,
          "unique_projects": 368
        },
        "risk": {
          "accuracy": 0.2236,
          "macro_precision": 0.3102,
          "macro_recall": 0.3003,
          "macro_f1": 0.2243,
          "confusion_matrix": [
            [
              2.5219,
              0.7428,
              1.5584,
              0.3606
            ],
            [
              1.6429,
              2.0418,
              0.5111,
              0.1484
            ],
            [
              1.2744,
              2.8638,
              0.8982,
              0.0545
            ],
            [
              2.7408,
              6.9885,
              3.42,
              0.9613
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      },
      "very_late": {
        "available": true,
        "cost": {
          "MAE": 25.417,
          "RMSE": 54.965,
          "R2": 0.3582,
          "MAPE": 148.323,
          "rows": 8159,
          "unique_projects": 652
        },
        "delay": {
          "MAE": 526.533,
          "RMSE": 703.212,
          "R2": 0.5584,
          "MAPE": 58.901,
          "rows": 8159,
          "unique_projects": 652
        },
        "risk": {
          "accuracy": 0.6581,
          "macro_precision": 0.4544,
          "macro_recall": 0.3899,
          "macro_f1": 0.4093,
          "confusion_matrix": [
            [
              0.9782,
              1.4576,
              1.4287,
              0.6854
            ],
            [
              0.4131,
              6.9719,
              5.4345,
              8.0924
            ],
            [
              0.214,
              4.6034,
              6.385,
              19.3936
            ],
            [
              0.4525,
              6.7259,
              23.7246,
              125.4794
            ]
          ],
          "labels": [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
          ]
        }
      }
    }
  }
}
```

### SHAP / importance

```json
{
  "cost": {
    "method": "mean_absolute_shap",
    "features": [
      {
        "feature": "expenditure_ratio",
        "importance": 0.295997
      },
      {
        "feature": "cost_escalation_percentage",
        "importance": 0.211542
      },
      {
        "feature": "sector_average_delay",
        "importance": 0.074688
      },
      {
        "feature": "planned_duration_days",
        "importance": 0.065101
      },
      {
        "feature": "approved_cost_cr",
        "importance": 0.059189
      },
      {
        "feature": "cumulative_expenditure_cr",
        "importance": 0.051566
      },
      {
        "feature": "elapsed_duration_days",
        "importance": 0.042977
      },
      {
        "feature": "implementing_agency",
        "importance": 0.032358
      },
      {
        "feature": "agency_average_cost_overrun",
        "importance": 0.031836
      },
      {
        "feature": "agency_cost_overrun_rate",
        "importance": 0.027045
      },
      {
        "feature": "sector",
        "importance": 0.022282
      },
      {
        "feature": "schedule_slippage_days",
        "importance": 0.016555
      },
      {
        "feature": "schedule_slippage_ratio",
        "importance": 0.016133
      },
      {
        "feature": "agency_delay_rate",
        "importance": 0.012478
      },
      {
        "feature": "revised_cost_cr",
        "importance": 0.012173
      },
      {
        "feature": "missingindicator_schedule_slippage_days",
        "importance": 0.006906
      },
      {
        "feature": "agency_average_delay",
        "importance": 0.006124
      },
      {
        "feature": "sector_average_cost_overrun",
        "importance": 0.004052
      },
      {
        "feature": "duration_ratio",
        "importance": 0.003881
      },
      {
        "feature": "project_size_category",
        "importance": 0.00379
      },
      {
        "feature": "missingindicator_planned_duration_days",
        "importance": 0.002069
      },
      {
        "feature": "missingindicator_duration_ratio",
        "importance": 0.000448
      },
      {
        "feature": "missingindicator_schedule_slippage_ratio",
        "importance": 0.000366
      },
      {
        "feature": "cost_growth_velocity_6m",
        "importance": 0.000318
      },
      {
        "feature": "missingindicator_cost_growth_velocity_3m",
        "importance": 7.8e-05
      },
      {
        "feature": "expected_progress_percentage",
        "importance": 3e-05
      },
      {
        "feature": "missingindicator_cost_growth_velocity_6m",
        "importance": 1.9e-05
      },
      {
        "feature": "cost_growth_velocity_3m",
        "importance": 0.0
      },
      {
        "feature": "cost_acceleration",
        "importance": 0.0
      },
      {
        "feature": "sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_expected_progress_percentage",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_revised_cost_cr",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_escalation_percentage",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_acceleration",
        "importance": 0.0
      }
    ]
  },
  "delay": {
    "method": "mean_absolute_shap",
    "features": [
      {
        "feature": "elapsed_duration_days",
        "importance": 0.303994
      },
      {
        "feature": "duration_ratio",
        "importance": 0.216681
      },
      {
        "feature": "sector_average_delay",
        "importance": 0.131048
      },
      {
        "feature": "planned_duration_days",
        "importance": 0.0625
      },
      {
        "feature": "schedule_slippage_days",
        "importance": 0.056246
      },
      {
        "feature": "sector",
        "importance": 0.051539
      },
      {
        "feature": "sector_average_cost_overrun",
        "importance": 0.034383
      },
      {
        "feature": "agency_average_delay",
        "importance": 0.021948
      },
      {
        "feature": "agency_cost_overrun_rate",
        "importance": 0.021571
      },
      {
        "feature": "agency_delay_rate",
        "importance": 0.01451
      },
      {
        "feature": "cost_escalation_percentage",
        "importance": 0.012383
      },
      {
        "feature": "approved_cost_cr",
        "importance": 0.011025
      },
      {
        "feature": "revised_cost_cr",
        "importance": 0.010934
      },
      {
        "feature": "implementing_agency",
        "importance": 0.010441
      },
      {
        "feature": "expenditure_ratio",
        "importance": 0.010134
      },
      {
        "feature": "expected_progress_percentage",
        "importance": 0.009281
      },
      {
        "feature": "agency_average_cost_overrun",
        "importance": 0.007463
      },
      {
        "feature": "schedule_slippage_ratio",
        "importance": 0.005779
      },
      {
        "feature": "cumulative_expenditure_cr",
        "importance": 0.003354
      },
      {
        "feature": "missingindicator_schedule_slippage_days",
        "importance": 0.002359
      },
      {
        "feature": "missingindicator_cost_growth_velocity_3m",
        "importance": 0.000656
      },
      {
        "feature": "missingindicator_cost_growth_velocity_6m",
        "importance": 0.00063
      },
      {
        "feature": "missingindicator_schedule_slippage_ratio",
        "importance": 0.000567
      },
      {
        "feature": "project_size_category",
        "importance": 0.000347
      },
      {
        "feature": "missingindicator_planned_duration_days",
        "importance": 0.000226
      },
      {
        "feature": "cost_growth_velocity_3m",
        "importance": 0.0
      },
      {
        "feature": "cost_growth_velocity_6m",
        "importance": 0.0
      },
      {
        "feature": "cost_acceleration",
        "importance": 0.0
      },
      {
        "feature": "sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_duration_ratio",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_expected_progress_percentage",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_revised_cost_cr",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_escalation_percentage",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_acceleration",
        "importance": 0.0
      }
    ]
  },
  "risk": {
    "method": "tree_feature_importance",
    "features": [
      {
        "feature": "sector",
        "importance": 0.137309
      },
      {
        "feature": "implementing_agency",
        "importance": 0.114011
      },
      {
        "feature": "duration_ratio",
        "importance": 0.0892
      },
      {
        "feature": "expected_progress_percentage",
        "importance": 0.069867
      },
      {
        "feature": "elapsed_duration_days",
        "importance": 0.064961
      },
      {
        "feature": "schedule_slippage_days",
        "importance": 0.063067
      },
      {
        "feature": "schedule_slippage_ratio",
        "importance": 0.04855
      },
      {
        "feature": "planned_duration_days",
        "importance": 0.047442
      },
      {
        "feature": "expenditure_ratio",
        "importance": 0.036498
      },
      {
        "feature": "agency_average_delay",
        "importance": 0.033269
      },
      {
        "feature": "approved_cost_cr",
        "importance": 0.033082
      },
      {
        "feature": "revised_cost_cr",
        "importance": 0.03246
      },
      {
        "feature": "sector_average_delay",
        "importance": 0.029998
      },
      {
        "feature": "agency_delay_rate",
        "importance": 0.028507
      },
      {
        "feature": "cumulative_expenditure_cr",
        "importance": 0.028257
      },
      {
        "feature": "agency_cost_overrun_rate",
        "importance": 0.02556
      },
      {
        "feature": "agency_average_cost_overrun",
        "importance": 0.024895
      },
      {
        "feature": "sector_average_cost_overrun",
        "importance": 0.022206
      },
      {
        "feature": "cost_escalation_percentage",
        "importance": 0.020283
      },
      {
        "feature": "project_size_category",
        "importance": 0.019516
      },
      {
        "feature": "cost_growth_velocity_6m",
        "importance": 0.004445
      },
      {
        "feature": "missingindicator_cost_acceleration",
        "importance": 0.004022
      },
      {
        "feature": "missingindicator_cost_growth_velocity_3m",
        "importance": 0.003843
      },
      {
        "feature": "cost_growth_velocity_3m",
        "importance": 0.003468
      },
      {
        "feature": "missingindicator_cost_growth_velocity_6m",
        "importance": 0.003321
      },
      {
        "feature": "cost_acceleration",
        "importance": 0.003048
      },
      {
        "feature": "missingindicator_schedule_slippage_days",
        "importance": 0.002069
      },
      {
        "feature": "missingindicator_duration_ratio",
        "importance": 0.002038
      },
      {
        "feature": "missingindicator_planned_duration_days",
        "importance": 0.001777
      },
      {
        "feature": "missingindicator_expected_progress_percentage",
        "importance": 0.001587
      },
      {
        "feature": "missingindicator_schedule_slippage_ratio",
        "importance": 0.001442
      },
      {
        "feature": "sector_delay_rate",
        "importance": 0.0
      },
      {
        "feature": "sector_cost_overrun_rate",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_revised_cost_cr",
        "importance": 0.0
      },
      {
        "feature": "missingindicator_cost_escalation_percentage",
        "importance": 0.0
      }
    ]
  }
}
```

Conclusion for this window: both cost and delay MAE improved.
Each ablation repeats the same training-only candidate selection protocol; the final holdout remains untouched. Trajectory and agency-prior effects are interpreted metric by metric and do not inherit the full model's overall improvement claim.

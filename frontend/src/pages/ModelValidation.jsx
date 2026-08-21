import React from 'react';

export default function ModelValidation() {
  const metrics = [
    { label: 'Cost Prediction MAE', value: 'Pending training' },
    { label: 'Delay Prediction Error', value: 'Pending training' },
    { label: 'Backtest Projects', value: 'Historical validation set' },
    { label: 'Explainability', value: 'SHAP enabled' },
  ];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold">AI Model Validation</h1>
      <p className="mt-2 text-gray-600">
        Historical backtesting dashboard showing prediction quality against actual project outcomes.
      </p>
      <div className="grid gap-4 mt-6 md:grid-cols-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="rounded-xl border p-4">
            <div className="text-sm text-gray-500">{metric.label}</div>
            <div className="mt-2 font-medium">{metric.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

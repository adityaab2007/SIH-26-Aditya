import { api } from '../services/api.js';
import { horizontalBars } from '../components/charts.js';

const fixed = (value, digits = 2) => Number(value || 0).toFixed(digits);
const escape = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const officialUrl = (value = '') => String(value).startsWith('https://paimana-proj.mospi.gov.in/') ? String(value) : '';

function yearOptions(years, selected) {
  return years.map((item) => `<option value="${item.year}" ${item.year === selected ? 'selected' : ''}>${item.year} · ${item.completed_projects} lifecycle projects</option>`).join('');
}

function predictionCard(prediction, actual = null) {
  const source = actual ? officialUrl(actual.source_url) : '';
  const range = prediction.expected_range;
  const confidenceText = prediction.confidence_calibration_status === 'not_calibrated_for_live_lifecycle_retrain'
    ? 'Not calibrated for this live retrain'
    : `${fixed(prediction.model_confidence_percentage, 1)}% · ${escape(prediction.confidence_calibration_status || 'unavailable').replaceAll('_', ' ')}`;
  return `<div class="model-grid">
    <section class="panel">
      <span class="kicker">Monthly lifecycle AI prediction generated first</span>
      <h2>${escape(prediction.project.project_name)}</h2>
      <div class="detail-financial">
        <div><span>Predicted cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div>
        <div><span>Predicted delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div>
        <div><span>Predicted risk</span><strong>${escape(prediction.predicted_risk)} · ${fixed(prediction.risk_probability_percentage, 1)}%</strong></div>
        <div><span>Prediction snapshot</span><strong>${escape(prediction.snapshot_date || 'Unavailable')}</strong></div>
        <div><span>Official history snapshots</span><strong>${prediction.history_snapshots || 1}</strong></div>
        <div><span>Confidence calibration</span><strong>${confidenceText}</strong></div>
        <div><span>Actual outcome sent yet?</span><strong>${prediction.audit.actual_outcomes_sent_to_browser ? 'Yes' : 'No'}</strong></div>
      </div>
      <h3>Lifecycle inputs visible to the model</h3>
      <div class="detail-financial">
        <div><span>Feature count</span><strong>${Object.keys(prediction.model_inputs || {}).length}</strong></div>
        <div><span>Approved cost</span><strong>₹${fixed(prediction.model_inputs.approved_cost_cr)} Cr</strong></div>
        <div><span>Revised cost</span><strong>₹${fixed(prediction.model_inputs.revised_cost_cr)} Cr</strong></div>
        <div><span>Expenditure ratio</span><strong>${fixed((prediction.model_inputs.expenditure_ratio || 0) * 100, 1)}%</strong></div>
        <div><span>Schedule slippage</span><strong>${fixed(prediction.model_inputs.schedule_slippage_days)} days</strong></div>
        <div><span>Duration ratio</span><strong>${fixed((prediction.model_inputs.duration_ratio || 0) * 100, 1)}%</strong></div>
        <div><span>Sector</span><strong>${escape(prediction.model_inputs.sector)}</strong></div>
        <div><span>Implementing agency</span><strong>${escape(prediction.model_inputs.implementing_agency)}</strong></div>
      </div>
      ${range ? `<div class="notice compact"><strong>Uncertainty range:</strong> Cost P10–P90 ${fixed(range.cost_overrun_percentage.p10)}% to ${fixed(range.cost_overrun_percentage.p90)}%; delay P10–P90 ${fixed(range.delay_days.p10)} to ${fixed(range.delay_days.p90)} days.</div>` : ''}
    </section>
    <section class="panel">
      <span class="kicker">Explainability</span>
      <h2>Why the lifecycle model predicted this</h2>
      ${horizontalBars(prediction.shap_explanation.map((factor) => ({ label: `${factor.feature} (${factor.direction})`, value: Math.abs(factor.impact) })), { format: (v) => fixed(v, 3) })}
      <div class="notice compact"><strong>Leakage audit:</strong> This project is excluded from the selected training years: ${prediction.audit.project_excluded_from_training ? 'YES' : 'NO'}.</div>
    </section>
  </div>
  ${actual ? `<section class="panel"><span class="kicker">Official outcome revealed after prediction</span><h2>Prediction vs actual</h2><div class="detail-financial"><div><span>AI cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div><div><span>Actual cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div><div><span>Absolute cost error</span><strong>${fixed(actual.cost_error_absolute_pp)} pp</strong></div><div><span>AI delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div><div><span>Actual delay</span><strong>${fixed(actual.actual_delay_days)} days</strong></div><div><span>Absolute delay error</span><strong>${fixed(actual.delay_error_absolute_days)} days</strong></div><div><span>AI / actual risk</span><strong>${escape(prediction.predicted_risk)} / ${escape(actual.actual_risk)}</strong></div><div><span>Recorded completion</span><strong>${escape(actual.completion_date)}</strong></div></div><div class="notice compact"><strong>Reveal audit:</strong> ${escape(actual.reveal_policy)}</div>${source ? `<a class="secondary-btn" href="${escape(source)}" target="_blank" rel="noopener noreferrer">Open official PAIMANA source</a>` : ''}</section>` : '<div class="notice compact"><strong>Actual outcome is still hidden.</strong> Click Reveal Actual Outcome only after the judge has seen the AI prediction.</div>'}`;
}

export async function ModelSimulationPage(root) {
  const catalog = await api.simulationVersions();
  if (!catalog.lifecycle_data_available) {
    root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1><p>Live retraining requires the official processed PAIMANA monthly lifecycle dataset.</p></div></header>
    <section class="panel"><div class="error-state">Official PAIMANA monthly lifecycle dataset is not available in this checkout.</div><p class="muted">Refresh the official archive separately, then rebuild <code>data/processed/paimana_monthly_snapshots.csv</code>. The Retrain button does not download or parse PAIMANA PDFs.</p><button class="primary-btn" disabled>Retrain Lifecycle Models Live</button></section>`;
    return;
  }
  const years = catalog.data_years || [];
  if (!years.length) throw new Error('No identity-verified PAIMANA lifecycle years are available.');

  const yearNumbers = years.map((item) => item.year);
  const defaultStart = yearNumbers[0];
  const preferredEnd = yearNumbers.filter((year) => year <= 2015).at(-1);
  const defaultEnd = preferredEnd || yearNumbers[Math.max(0, Math.floor(yearNumbers.length / 2) - 1)];

  root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1><p>Choose a historical training range and retrain the monthly lifecycle cost, delay, and risk models from scratch. Then test an official project that completed only after that cutoff.</p></div></header>
  <div class="notice"><strong>Leakage rule:</strong> Algorithm selection happens inside the selected training period. Projects completed after the training cutoff are held out from fitting and are used only for future evaluation and judge-selected prediction.</div>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 1</span><h2>Choose training years and retrain lifecycle models</h2></div></div>
    <div class="filters">
      <label>Training start year<select id="custom-start">${yearOptions(years, defaultStart)}</select></label>
      <label>Training end year<select id="custom-end">${yearOptions(years, defaultEnd)}</select></label>
      <button class="primary-btn" id="custom-train">Retrain Lifecycle Models Live</button>
    </div>
    <div id="training-receipt" class="notice compact">No lifecycle model has been trained in this demo session yet.</div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 2</span><h2>Judge chooses an unseen future project</h2></div></div>
    <div class="filters">
      <label>Held-out completion year<select id="custom-test-year" disabled><option>Retrain first</option></select></label>
      <label>Official held-out project<select id="custom-project" disabled><option>Select a test year first</option></select></label>
      <button class="secondary-btn" id="random-project" disabled>Pick Random Unseen Project</button>
      <button class="primary-btn" id="custom-predict" disabled>Generate Lifecycle Prediction</button>
      <button class="secondary-btn" id="custom-reveal" disabled>Reveal Actual Outcome</button>
    </div>
    <div id="held-out-note" class="notice compact">After retraining, only projects completed after the selected training end year will be offered here.</div>
  </section>
  <div id="custom-output"></div>`;

  const start = root.querySelector('#custom-start');
  const end = root.querySelector('#custom-end');
  const trainButton = root.querySelector('#custom-train');
  const receipt = root.querySelector('#training-receipt');
  const testYear = root.querySelector('#custom-test-year');
  const project = root.querySelector('#custom-project');
  const randomButton = root.querySelector('#random-project');
  const predictButton = root.querySelector('#custom-predict');
  const revealButton = root.querySelector('#custom-reveal');
  const heldOutNote = root.querySelector('#held-out-note');
  const output = root.querySelector('#custom-output');

  let session = null;
  let projectRows = [];
  let prediction = null;
  let actual = null;

  const resetPrediction = () => {
    prediction = null;
    actual = null;
    revealButton.disabled = true;
    output.innerHTML = '';
  };

  const loadProjects = async () => {
    if (!session || !testYear.value) return;
    resetPrediction();
    project.disabled = true;
    predictButton.disabled = true;
    randomButton.disabled = true;
    heldOutNote.innerHTML = '<div class="loading">Loading held-out official lifecycle projects…</div>';
    try {
      const response = await api.customSimulationProjects(session.session_id, Number(testYear.value));
      projectRows = response.items;
      project.innerHTML = projectRows.map((row) => `<option value="${row.record_index}">${escape(row.project_id)} · ${escape(row.project_name)}</option>`).join('');
      project.disabled = !projectRows.length;
      predictButton.disabled = !projectRows.length;
      randomButton.disabled = !projectRows.length;
      heldOutNote.innerHTML = `<strong>${projectRows.length} held-out projects available for ${escape(response.year)}.</strong> ${escape(response.note)} No actual cost, delay, completion date, or final expenditure has been sent to this page.`;
    } catch (error) {
      projectRows = [];
      project.innerHTML = '<option>No projects available</option>';
      heldOutNote.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
    }
  };

  const generatePrediction = async () => {
    if (!session || project.disabled) return;
    resetPrediction();
    predictButton.disabled = true;
    output.innerHTML = '<div class="loading">Generating prediction from the freshly trained monthly lifecycle model…</div>';
    try {
      prediction = await api.predictCustomSimulation(session.session_id, Number(project.value));
      revealButton.disabled = false;
      output.innerHTML = predictionCard(prediction);
    } catch (error) {
      output.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
    } finally {
      predictButton.disabled = false;
    }
  };

  trainButton.addEventListener('click', async () => {
    resetPrediction();
    const startYear = Number(start.value);
    const endYear = Number(end.value);
    if (startYear > endYear) {
      receipt.innerHTML = '<div class="error-state">Training start year cannot be after training end year.</div>';
      return;
    }
    trainButton.disabled = true;
    testYear.disabled = true;
    project.disabled = true;
    predictButton.disabled = true;
    randomButton.disabled = true;
    receipt.innerHTML = '<div class="loading">Building selected PAIMANA lifecycle cohort → auditing features → selecting cost/delay regressors on internal temporal validation → fitting final cost, delay, and risk models → evaluating future holdout…</div>';
    try {
      const registryRun = await api.retrainModel(startYear, endYear);
      session = await api.trainCustomSimulation(startYear, endYear);
      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} held-out projects</option>`).join('');
      testYear.disabled = !eligible.length;
      const quality = registryRun.metrics.metadata.feature_quality || {};
      const baseline = registryRun.baseline_comparison || {};
      const algorithms = registryRun.selected_algorithms || {};
      receipt.innerHTML = `<strong>${escape(registryRun.model_version)} retrained from scratch.</strong> Training: ${escape(registryRun.training_years)} · internal validation: ${escape(registryRun.internal_validation_year)} · untouched future holdout: ${escape(registryRun.testing_years)}. <strong>Lifecycle features:</strong> ${registryRun.feature_count} retained · ${quality.removed_invalid_feature_count || 0} rejected by the selected window audit. <strong>Selected models:</strong> cost ${escape(algorithms.cost || 'unknown')} · delay ${escape(algorithms.delay || 'unknown')} · risk Random Forest. <strong>Fresh holdout metrics:</strong> cost MAE ${fixed(registryRun.metrics.cost_model.MAE)} pp · delay MAE ${fixed(registryRun.metrics.delay_model.MAE)} days · risk macro-F1 ${fixed(registryRun.metrics.risk_model.macro_f1 * 100, 1)}%. <strong>Five-feature benchmark:</strong> cost MAE ${fixed(baseline.cost_mae)} pp · delay MAE ${fixed(baseline.delay_mae)} days · risk macro-F1 ${fixed((baseline.risk_macro_f1 || 0) * 100, 1)}%. <strong>Feature quality:</strong> ${fixed(quality.data_quality_score, 1)}%.<br><strong>Leakage guard:</strong> ${escape(session.leakage_guard)} Browser received actual held-out outcomes: <strong>${session.actual_outcomes_sent_to_browser ? 'YES' : 'NO'}</strong>.`;
      if (eligible.length) await loadProjects();
    } catch (error) {
      session = null;
      receipt.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
    } finally {
      trainButton.disabled = false;
    }
  });

  testYear.addEventListener('change', loadProjects);
  project.addEventListener('change', resetPrediction);
  predictButton.addEventListener('click', generatePrediction);
  randomButton.addEventListener('click', async () => {
    if (!projectRows.length) return;
    const chosen = projectRows[Math.floor(Math.random() * projectRows.length)];
    project.value = String(chosen.record_index);
    await generatePrediction();
  });
  revealButton.addEventListener('click', async () => {
    if (!session || !prediction) return;
    revealButton.disabled = true;
    try {
      actual = await api.revealCustomSimulation(session.session_id, prediction.record_index);
      output.innerHTML = predictionCard(prediction, actual);
    } catch (error) {
      output.innerHTML += `<div class="error-state">${escape(error.message)}</div>`;
      revealButton.disabled = false;
    }
  });
}

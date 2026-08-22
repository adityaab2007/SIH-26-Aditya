import { api } from '../services/api.js';
import { horizontalBars } from '../components/charts.js';

const fixed = (value, digits = 2) => Number(value || 0).toFixed(digits);
const escape = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const officialUrl = (value = '') => String(value).startsWith('https://paimana-proj.mospi.gov.in/') ? String(value) : '';

function yearOptions(years, selected) {
  return years.map((item) => `<option value="${item.year}" ${item.year === selected ? 'selected' : ''}>${item.year} · ${item.completed_projects} completed projects</option>`).join('');
}

function predictionCard(prediction, actual = null) {
  const source = actual ? officialUrl(actual.source_url) : '';
  return `<div class="model-grid">
    <section class="panel">
      <span class="kicker">AI prediction generated first</span>
      <h2>${escape(prediction.project.project_name)}</h2>
      <div class="detail-financial">
        <div><span>Predicted cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div>
        <div><span>Predicted delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div>
        <div><span>Predicted risk</span><strong>${escape(prediction.predicted_risk)}</strong></div>
        <div><span>Actual outcome sent yet?</span><strong>${prediction.audit.actual_outcomes_sent_to_browser ? 'Yes' : 'No'}</strong></div>
      </div>
      <h3>Inputs visible to the model</h3>
      <div class="detail-financial">
        <div><span>Approved cost</span><strong>₹${fixed(prediction.model_inputs.approved_cost_cr)} Cr</strong></div>
        <div><span>Sector</span><strong>${escape(prediction.model_inputs.sector)}</strong></div>
        <div><span>Implementing agency</span><strong>${escape(prediction.model_inputs.implementing_agency || 'Not reported')}</strong></div>
        <div><span>Planned commissioning year</span><strong>${escape(prediction.model_inputs.planned_commissioning_year)}</strong></div>
      </div>
    </section>
    <section class="panel">
      <span class="kicker">Explainability</span>
      <h2>Why the model predicted this</h2>
      ${horizontalBars(prediction.shap_explanation.map((factor) => ({ label: `${factor.feature} (${factor.direction})`, value: Math.abs(factor.impact) })), { format: (v) => fixed(v, 3) })}
      <div class="notice compact"><strong>Leakage audit:</strong> This project is excluded from the training years: ${prediction.audit.project_excluded_from_training ? 'YES' : 'NO'}.</div>
    </section>
  </div>
  ${actual ? `<section class="panel"><span class="kicker">Official outcome revealed after prediction</span><h2>Prediction vs actual</h2><div class="detail-financial"><div><span>AI cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div><div><span>Actual cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div><div><span>Absolute cost error</span><strong>${fixed(actual.cost_error_absolute_pp)} pp</strong></div><div><span>AI delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div><div><span>Actual delay</span><strong>${fixed(actual.actual_delay_days)} days</strong></div><div><span>Absolute delay error</span><strong>${fixed(actual.delay_error_absolute_days)} days</strong></div><div><span>AI / actual risk</span><strong>${escape(prediction.predicted_risk)} / ${escape(actual.actual_risk)}</strong></div><div><span>Recorded completion</span><strong>${escape(actual.completion_date)}</strong></div></div><div class="notice compact"><strong>Reveal audit:</strong> ${escape(actual.reveal_policy)}</div>${source ? `<a class="secondary-btn" href="${escape(source)}" target="_blank" rel="noopener noreferrer">Open official PAIMANA source</a>` : ''}</section>` : '<div class="notice compact"><strong>Actual outcome is still hidden.</strong> Click Reveal Actual Outcome only after the judge has seen the AI prediction.</div>'}`;
}

export async function ModelSimulationPage(root) {
  const catalog = await api.simulationVersions();
  const years = catalog.data_years || [];
  if (!years.length) throw new Error('No official completed-project years are available.');

  const yearNumbers = years.map((item) => item.year);
  const defaultStart = yearNumbers[0];
  const preferredEnd = yearNumbers.filter((year) => year <= 2015).at(-1);
  const defaultEnd = preferredEnd || yearNumbers[Math.max(0, Math.floor(yearNumbers.length / 2) - 1)];

  root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical backtest</span><h1>Live Model Verification</h1><p>Choose the years used for training, retrain the model live, then test any official project from a later year. The browser does not receive the actual outcome until you reveal it.</p></div></header>
  <div class="notice"><strong>Why later years only?</strong> A valid historical forecast must train on the past and test on the future. Projects completed during or before the training cutoff cannot be used as the judge-selected test case.</div>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 1</span><h2>Choose training years and retrain</h2></div></div>
    <div class="filters">
      <label>Training start year<select id="custom-start">${yearOptions(years, defaultStart)}</select></label>
      <label>Training end year<select id="custom-end">${yearOptions(years, defaultEnd)}</select></label>
      <button class="primary-btn" id="custom-train">Retrain Model Live</button>
    </div>
    <div id="training-receipt" class="notice compact">No custom model has been trained in this demo session yet.</div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 2</span><h2>Judge chooses an unseen year and project</h2></div></div>
    <div class="filters">
      <label>Held-out test year<select id="custom-test-year" disabled><option>Retrain first</option></select></label>
      <label>Official held-out project<select id="custom-project" disabled><option>Select a test year first</option></select></label>
      <button class="secondary-btn" id="random-project" disabled>Pick Random Unseen Project</button>
      <button class="primary-btn" id="custom-predict" disabled>Generate AI Prediction</button>
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
    heldOutNote.innerHTML = '<div class="loading">Loading held-out official projects…</div>';
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
    output.innerHTML = '<div class="loading">Generating prediction using the freshly trained model…</div>';
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
    receipt.innerHTML = '<div class="loading">Loading official PAIMANA data → training models → evaluating future unseen projects → generating metrics…</div>';
    try {
      const registryRun = await api.retrainModel(startYear, endYear);
      session = await api.trainCustomSimulation(startYear, endYear);
      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} held-out projects</option>`).join('');
      testYear.disabled = !eligible.length;
      receipt.innerHTML = `<strong>Model ${escape(registryRun.model_version)} trained and registered.</strong> Training: ${escape(registryRun.training_years)} · testing: ${escape(registryRun.testing_years)}. <strong>Fresh metrics:</strong> cost MAE ${fixed(registryRun.metrics.cost_model.MAE)} pp · delay MAE ${fixed(registryRun.metrics.delay_model.MAE_days)} days · risk F1 ${fixed(registryRun.metrics.risk_model.f1 * 100, 1)}%. <strong>Rolling validation:</strong> ${registryRun.rolling_validation.fold_count} folds · average cost MAE ${fixed(registryRun.rolling_validation.average_cost_mae)} pp · average delay MAE ${fixed(registryRun.rolling_validation.average_delay_mae_days)} days. <a href="#/prediction-accuracy">Open dynamic Prediction Accuracy</a><br><strong>Leakage guard:</strong> ${escape(session.leakage_guard)} Browser received actual held-out outcomes: <strong>${session.actual_outcomes_sent_to_browser ? 'YES' : 'NO'}</strong>.`;
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

import { api } from '../services/api.js';
import { horizontalBars } from '../components/charts.js';

const missing = (value) => value === null || value === undefined || value === '' || (typeof value === 'number' && !Number.isFinite(value));
const fixed = (value, digits = 2) => missing(value) ? 'N/A' : Number(value).toFixed(digits);
const ratioPercent = (value, digits = 1) => missing(value) ? 'N/A' : `${(Number(value) * 100).toFixed(digits)}%`;
const fractionPercent = (value, digits = 1) => missing(value) ? 'N/A' : `${(Number(value) * 100).toFixed(digits)}%`;
const text = (value, fallback = 'Not reported') => missing(value) ? fallback : String(value);
const escape = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const officialUrl = (value = '') => String(value).startsWith('https://paimana-proj.mospi.gov.in/') ? String(value) : '';
const shortFingerprint = (value) => value ? String(value).replace('sha256:', '').slice(0, 12) : 'Not recorded';

function yearOptions(years, selected) {
  return years.map((item) => `<option value="${item.year}" ${item.year === selected ? 'selected' : ''}>${item.year} · ${item.completed_projects} lifecycle projects</option>`).join('');
}

function predictionCard(prediction, actual = null) {
  const source = actual ? officialUrl(actual.source_url) : '';
  const range = prediction.expected_range;
  const confidenceText = prediction.confidence_calibration_status === 'not_calibrated_for_live_lifecycle_retrain'
    ? 'Not calibrated for this live retrain'
    : `${fixed(prediction.model_confidence_percentage, 1)}% · ${escape(prediction.confidence_calibration_status || 'unavailable').replaceAll('_', ' ')}`;
  const factors = Array.isArray(prediction.shap_explanation) ? prediction.shap_explanation : [];
  const explanation = factors.length
    ? horizontalBars(factors.map((factor) => ({ label: `${factor.feature} (${factor.direction})`, value: Math.abs(factor.impact) })), { format: (v) => fixed(v, 3) })
    : '<p class="muted">No project-specific explanation factors were generated.</p>';
  return `<div class="notice compact"><strong>Run identity:</strong> ${escape(prediction.run_id || 'Not recorded')} · <strong>dataset:</strong> ${escape(shortFingerprint(prediction.dataset_fingerprint))}</div>
  <div class="model-grid">
    <section class="panel">
      <span class="kicker">Monthly lifecycle AI prediction generated first</span>
      <h2>${escape(text(prediction.project.project_name))}</h2>
      <div class="detail-financial">
        <div><span>Predicted cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div>
        <div><span>Predicted delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div>
        <div><span>Predicted risk</span><strong>${escape(text(prediction.predicted_risk))} · ${fixed(prediction.risk_probability_percentage, 1)}%</strong></div>
        <div><span>Prediction snapshot</span><strong>${escape(text(prediction.snapshot_date, 'Unavailable'))}</strong></div>
        <div><span>Official history snapshots</span><strong>${missing(prediction.history_snapshots) ? 'N/A' : prediction.history_snapshots}</strong></div>
        <div><span>Confidence calibration</span><strong>${confidenceText}</strong></div>
        <div><span>Actual outcome sent yet?</span><strong>${prediction.audit.actual_outcomes_sent_to_browser ? 'Yes' : 'No'}</strong></div>
      </div>
      <h3>Lifecycle inputs visible to the model</h3>
      <div class="detail-financial">
        <div><span>Feature count</span><strong>${Object.keys(prediction.model_inputs || {}).length}</strong></div>
        <div><span>Approved cost</span><strong>${missing(prediction.model_inputs.approved_cost_cr) ? 'N/A' : `₹${fixed(prediction.model_inputs.approved_cost_cr)} Cr`}</strong></div>
        <div><span>Revised cost</span><strong>${missing(prediction.model_inputs.revised_cost_cr) ? 'N/A' : `₹${fixed(prediction.model_inputs.revised_cost_cr)} Cr`}</strong></div>
        <div><span>Expenditure ratio</span><strong>${ratioPercent(prediction.model_inputs.expenditure_ratio)}</strong></div>
        <div><span>Schedule slippage</span><strong>${missing(prediction.model_inputs.schedule_slippage_days) ? 'N/A' : `${fixed(prediction.model_inputs.schedule_slippage_days)} days`}</strong></div>
        <div><span>Duration ratio</span><strong>${ratioPercent(prediction.model_inputs.duration_ratio)}</strong></div>
        <div><span>Sector</span><strong>${escape(text(prediction.model_inputs.sector))}</strong></div>
        <div><span>Implementing agency</span><strong>${escape(text(prediction.model_inputs.implementing_agency))}</strong></div>
      </div>
      ${range ? `<div class="notice compact"><strong>Uncertainty range:</strong> Cost P10–P90 ${fixed(range.cost_overrun_percentage?.p10)}% to ${fixed(range.cost_overrun_percentage?.p90)}%; delay P10–P90 ${fixed(range.delay_days?.p10)} to ${fixed(range.delay_days?.p90)} days.</div>` : ''}
    </section>
    <section class="panel">
      <span class="kicker">Explainability</span>
      <h2>Why the lifecycle model predicted this</h2>
      ${explanation}
      <div class="notice compact"><strong>Leakage audit:</strong> This project is excluded from the selected training years: ${prediction.audit.project_excluded_from_training ? 'YES' : 'NO'}.</div>
    </section>
  </div>
  ${actual ? `<section class="panel"><span class="kicker">Official outcome revealed after prediction</span><h2>Prediction vs actual</h2><div class="detail-financial"><div><span>AI cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div><div><span>Actual cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div><div><span>Absolute cost error</span><strong>${fixed(actual.cost_error_absolute_pp)} pp</strong></div><div><span>AI delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div><div><span>Actual delay</span><strong>${fixed(actual.actual_delay_days)} days</strong></div><div><span>Absolute delay error</span><strong>${fixed(actual.delay_error_absolute_days)} days</strong></div><div><span>AI / actual risk</span><strong>${escape(text(prediction.predicted_risk))} / ${escape(text(actual.actual_risk))}</strong></div><div><span>Recorded completion</span><strong>${escape(text(actual.completion_date))}</strong></div></div><div class="notice compact"><strong>Reveal audit:</strong> ${escape(actual.reveal_policy)} · run ${escape(actual.run_id || 'Not recorded')}</div>${source ? `<a class="secondary-btn" href="${escape(source)}" target="_blank" rel="noopener noreferrer">Open official PAIMANA source</a>` : ''}</section>` : '<div class="notice compact"><strong>Actual outcome is still hidden.</strong> Click Reveal Actual Outcome only after the judge has seen the AI prediction.</div>'}`;
}

function specialistPredictionCard(prediction, actual = null, convergence = null) {
  if (!prediction) return '';
  const cost = prediction.cost || {};
  const delay = prediction.delay || {};
  const global = prediction.global || {};
  const specialistCostError = actual ? Math.abs(Number(cost.predicted_final_overrun_percentage) - Number(actual.actual_cost_overrun)) : null;
  const specialistDelayError = actual ? Math.abs(Number(delay.predicted_final_delay_days) - Number(actual.actual_delay_days)) : null;
  const globalCostError = actual && global.cost ? Math.abs(Number(global.cost.predicted_final_overrun_percentage) - Number(actual.actual_cost_overrun)) : null;
  const globalDelayError = actual && global.delay ? Math.abs(Number(global.delay.predicted_final_delay_days) - Number(actual.actual_delay_days)) : null;
  const closer = specialistCostError == null || globalCostError == null ? 'N/A' : (specialistCostError <= globalCostError ? 'Lifecycle specialist' : 'Global lifecycle model');
  return `<section class="panel"><span class="kicker">Experiment 4 · one routed specialist</span><h2>Global vs lifecycle specialist</h2><div class="detail-financial"><div><span>Lifecycle stage</span><strong>${escape(text(prediction.lifecycle_stage))}</strong></div><div><span>Lifecycle percentage</span><strong>${fixed(prediction.lifecycle_percentage, 1)}%</strong></div><div><span>Global cost</span><strong>${fixed(global.cost?.predicted_final_overrun_percentage)}%</strong></div><div><span>Specialist cost</span><strong>${fixed(cost.predicted_final_overrun_percentage)}%</strong></div><div><span>Global delay</span><strong>${fixed(global.delay?.predicted_final_delay_days)} days</strong></div><div><span>Specialist delay</span><strong>${fixed(delay.predicted_final_delay_days)} days</strong></div><div><span>Cost algorithm</span><strong>${escape(text(cost.algorithm))}</strong></div><div><span>Delay algorithm</span><strong>${escape(text(delay.algorithm))}</strong></div></div>${prediction.fallback_to_global ? `<div class="notice compact"><strong>Global fallback:</strong> ${escape(prediction.fallback_reason || 'Specialist unavailable')}</div>` : ''}${actual ? `<div class="notice compact"><strong>Actual outcome revealed.</strong> Global cost error: ${fixed(globalCostError)} pp · specialist cost error: ${fixed(specialistCostError)} pp · global delay error: ${fixed(globalDelayError)} days · specialist delay error: ${fixed(specialistDelayError)} days. <strong>Closer on cost: ${escape(closer)}</strong></div>` : '<div class="notice compact"><strong>Actual outcome remains hidden</strong> until the existing Reveal Actual Outcome action.</div>'}${convergence ? `<h3>Forecast convergence from real snapshots</h3><div class="table-wrap"><table class="data-table compact-table"><thead><tr><th>Date</th><th>Lifecycle</th><th>Stage</th><th>Cost forecast</th><th>Delay forecast</th><th>Model</th></tr></thead><tbody>${(convergence.items || []).map((item) => `<tr><td>${escape(item.snapshot_date)}</td><td>${fixed(item.lifecycle_percentage, 1)}%</td><td>${escape(text(item.lifecycle_stage))}</td><td>${fixed(item.predicted_final_cost_overrun)}%</td><td>${fixed(item.predicted_final_delay)} days</td><td>${escape(text(item.specialist_model))}</td></tr>`).join('')}</tbody></table></div>` : ''}</section>`;
}

function trainingReceipt(registryRun, session = null, restored = false) {
  if (!registryRun) return 'No lifecycle model has been trained in this browser session yet.';
  const quality = registryRun.metrics?.metadata?.feature_quality || {};
  const baseline = registryRun.baseline_comparison || {};
  const algorithms = registryRun.selected_algorithms || {};
  const balanced = registryRun.balanced_stage_summary || registryRun.metrics?.metadata?.balanced_stage_summary || {};
  const sessionAudit = session ? ` <strong>Leakage guard:</strong> ${escape(session.leakage_guard)} Browser received actual held-out outcomes: <strong>${session.actual_outcomes_sent_to_browser ? 'YES' : 'NO'}</strong>.` : '';
  const provenance = registryRun.run_id ? ` <strong>Run ID:</strong> ${escape(registryRun.run_id)} · <strong>dataset:</strong> ${escape(shortFingerprint(registryRun.dataset_fingerprint))}.` : '';
  return `${restored ? '<strong>Restored active browser-session run.</strong> ' : ''}<strong>${escape(registryRun.model_version)} retrained from scratch.</strong> Training: ${escape(registryRun.training_years)} · internal validation: ${escape(registryRun.internal_validation_year)} · untouched future holdout: ${escape(registryRun.testing_years)}. <strong>Lifecycle features:</strong> ${registryRun.feature_count} retained · ${quality.removed_invalid_feature_count ?? 'N/A'} rejected by the selected window audit. <strong>Selected models:</strong> cost ${escape(algorithms.cost || 'unknown')} · delay ${escape(algorithms.delay || 'unknown')} · risk Random Forest. <strong>Fresh holdout metrics:</strong> cost MAE ${fixed(registryRun.metrics?.cost_model?.MAE)} pp · delay MAE ${fixed(registryRun.metrics?.delay_model?.MAE)} days · risk macro-F1 ${fractionPercent(registryRun.metrics?.risk_model?.macro_f1)}. <strong>Equal-stage diagnostic:</strong> cost MAE ${fixed(balanced.cost_mae)} pp · delay MAE ${fixed(balanced.delay_mae)} days · risk macro-F1 ${fractionPercent(balanced.risk_macro_f1)}. <strong>Five-feature benchmark:</strong> cost MAE ${fixed(baseline.cost_mae)} pp · delay MAE ${fixed(baseline.delay_mae)} days · risk macro-F1 ${fractionPercent(baseline.risk_macro_f1)}. <strong>Feature quality:</strong> ${fixed(quality.data_quality_score, 1)}%.${provenance}${sessionAudit}<br><a class="secondary-btn" href="#/prediction-accuracy">View Prediction Accuracy</a>`;
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

  const savedRun = api.getActiveLifecycleRun();
  const yearNumbers = years.map((item) => item.year);
  const savedStart = Number(savedRun?.start_year);
  const savedEnd = Number(savedRun?.end_year);
  const defaultStart = yearNumbers.includes(savedStart) ? savedStart : yearNumbers[0];
  const preferredEnd = yearNumbers.filter((year) => year <= 2015).at(-1);
  const fallbackEnd = preferredEnd || yearNumbers[Math.max(0, Math.floor(yearNumbers.length / 2) - 1)];
  const defaultEnd = yearNumbers.includes(savedEnd) ? savedEnd : fallbackEnd;

  root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1><p>Choose a historical training range and retrain the monthly lifecycle cost, delay, and risk models from scratch. The active run is kept for this browser session when you navigate to other pages.</p></div></header>
  <div class="notice"><strong>Leakage rule:</strong> Algorithm selection happens inside the selected training period. Projects completed after the training cutoff are held out from fitting and are used only for future evaluation and judge-selected prediction.</div>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 1</span><h2>Choose training years and retrain lifecycle models</h2></div></div>
    <div class="filters">
      <label>Training start year<select id="custom-start">${yearOptions(years, defaultStart)}</select></label>
      <label>Training end year<select id="custom-end">${yearOptions(years, defaultEnd)}</select></label>
      <button class="primary-btn" id="custom-train">Retrain Lifecycle Models Live</button>
    </div>
    <div id="training-receipt" class="notice compact">No lifecycle model has been trained in this browser session yet.</div>
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
  <section class="panel"><div class="panel-head"><div><span class="kicker">Experiment 4</span><h2>Lifecycle Specialist Comparison</h2></div></div><p class="muted">Train the experiment separately, then compare one routed specialist with the global lifecycle model for the selected snapshot. Actual outcomes remain hidden until the existing reveal action.</p><button class="secondary-btn" id="specialist-train">Train Lifecycle Specialists</button><div id="specialist-output" class="notice compact">No specialist experiment is active for this training window.</div></section>
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
  const specialistTrainButton = root.querySelector('#specialist-train');
  const specialistOutput = root.querySelector('#specialist-output');

  let session = null;
  let projectRows = [];
  let prediction = null;
  let actual = null;
  let specialistPrediction = null;
  let specialistWindow = null;
  let convergence = null;

  specialistTrainButton.addEventListener('click', async () => {
    specialistTrainButton.disabled = true;
    specialistOutput.textContent = 'Training eight experiment-only cost/delay specialists…';
    try {
      const result = await api.lifecycleSpecialistsRetrain(Number(start.value), Number(end.value));
      specialistWindow = `${Number(start.value)}_${Number(end.value)}`;
      const cost = result.specialist_overall?.cost?.MAE;
      const delay = result.specialist_overall?.delay?.MAE;
      specialistOutput.innerHTML = `<strong>Experiment 4 trained.</strong> Routed holdout cost MAE: ${fixed(cost)} pp · delay MAE: ${fixed(delay)} days. The global model remains the production default.`;
    } catch (error) {
      specialistOutput.innerHTML = `<strong>Experiment 4 unavailable:</strong> ${escape(error.message)}`;
    } finally {
      specialistTrainButton.disabled = false;
    }
  });

  const resetPrediction = () => {
    prediction = null;
    actual = null;
    specialistPrediction = null;
    convergence = null;
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
      if (session.run_id && response.run_id !== session.run_id) throw new Error('Held-out project response belongs to a different model run. Retrain this range.');
      if (session.dataset_fingerprint && response.dataset_fingerprint !== session.dataset_fingerprint) throw new Error('Held-out project response belongs to a different dataset snapshot. Retrain this range.');
      projectRows = response.items;
      project.innerHTML = projectRows.map((row) => `<option value="${row.record_index}">${escape(row.project_id)} · ${escape(row.project_name)}</option>`).join('');
      project.disabled = !projectRows.length;
      predictButton.disabled = !projectRows.length;
      randomButton.disabled = !projectRows.length;
      heldOutNote.innerHTML = `<strong>${projectRows.length} held-out projects available for ${escape(response.year)}.</strong> ${escape(response.note)} No actual cost, delay, completion date, or final expenditure has been sent to this page.`;
    } catch (error) {
      projectRows = [];
      project.innerHTML = '<option>No projects available</option>';
      heldOutNote.innerHTML = `<div class="error-state">${escape(error.message)}</div><p class="muted">If the backend was restarted or this run was replaced, retrain this saved range to create a new judge session.</p>`;
    }
  };

  const generatePrediction = async () => {
    if (!session || project.disabled) return;
    resetPrediction();
    predictButton.disabled = true;
    output.innerHTML = '<div class="loading">Generating prediction from the freshly trained monthly lifecycle model…</div>';
    try {
      prediction = await api.predictCustomSimulation(session.session_id, Number(project.value));
      if (session.run_id && prediction.run_id !== session.run_id) throw new Error('Prediction belongs to a different model run. Retrain this range.');
      if (session.dataset_fingerprint && prediction.dataset_fingerprint !== session.dataset_fingerprint) throw new Error('Prediction belongs to a different dataset snapshot. Retrain this range.');
      revealButton.disabled = false;
      const selectedProject = projectRows.find((row) => String(row.record_index) === String(project.value));
      if (specialistWindow && selectedProject?.project_id) {
        try {
          specialistPrediction = await api.lifecycleSpecialistForecast(selectedProject.project_id, specialistWindow);
          specialistPrediction.global = { cost: { predicted_final_overrun_percentage: prediction.predicted_cost_overrun }, delay: { predicted_final_delay_days: prediction.predicted_delay_days } };
          convergence = await api.lifecycleSpecialistsConvergence(selectedProject.project_id, specialistWindow);
        } catch (_) {
          specialistPrediction = null;
          convergence = null;
        }
      }
      output.innerHTML = predictionCard(prediction) + specialistPredictionCard(specialistPrediction, null, convergence);
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
      const activeBase = {
        window: registryRun.window,
        model_version: registryRun.model_version,
        run_id: registryRun.run_id,
        dataset_fingerprint: registryRun.dataset_fingerprint,
        start_year: startYear,
        end_year: endYear,
        receipt: registryRun,
      };
      api.setActiveLifecycleRun(activeBase);
      receipt.innerHTML = `${trainingReceipt(registryRun)}<div class="loading">Preparing judge-controlled held-out project session…</div>`;

      try {
        session = await api.trainCustomSimulation(startYear, endYear, registryRun.run_id);
        if (registryRun.run_id && session.run_id !== registryRun.run_id) throw new Error('Judge session did not bind to the exact retrained model run. Retrain again.');
        if (registryRun.dataset_fingerprint && session.dataset_fingerprint !== registryRun.dataset_fingerprint) throw new Error('Judge session dataset fingerprint does not match the retrained model. Retrain again.');
        api.setActiveLifecycleRun({ ...activeBase, session });
      } catch (sessionError) {
        session = null;
        receipt.innerHTML = `${trainingReceipt(registryRun)}<div class="error-state">The model was trained and saved, but the judge session could not be created: ${escape(sessionError.message)}</div>`;
        return;
      }

      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} held-out projects</option>`).join('');
      testYear.disabled = !eligible.length;
      receipt.innerHTML = trainingReceipt(registryRun, session);
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
      if (session.run_id && actual.run_id !== session.run_id) throw new Error('Reveal response belongs to a different model run.');
      if (session.dataset_fingerprint && actual.dataset_fingerprint !== session.dataset_fingerprint) throw new Error('Reveal response belongs to a different dataset snapshot.');
      const selectedProject = projectRows.find((row) => String(row.record_index) === String(project.value));
      if (specialistWindow && selectedProject?.project_id) {
        try { convergence = await api.lifecycleSpecialistsConvergence(selectedProject.project_id, specialistWindow, true); } catch (_) { /* optional reveal detail */ }
      }
      output.innerHTML = predictionCard(prediction, actual) + specialistPredictionCard(specialistPrediction, actual, convergence);
    } catch (error) {
      output.innerHTML += `<div class="error-state">${escape(error.message)}</div>`;
      revealButton.disabled = false;
    }
  });

  if (savedRun?.receipt && savedRun.start_year === defaultStart && savedRun.end_year === defaultEnd) {
    receipt.innerHTML = trainingReceipt(savedRun.receipt, savedRun.session || null, true);
    if (savedRun.session?.session_id) {
      session = savedRun.session;
      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} held-out projects</option>`).join('');
      testYear.disabled = !eligible.length;
      if (eligible.length) await loadProjects();
    } else {
      heldOutNote.innerHTML = '<strong>The trained model is still the active browser-session run.</strong> Retrain only if you need a new judge session or a different year range.';
    }
  }
}

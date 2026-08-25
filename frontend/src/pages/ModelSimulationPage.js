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

function improvementLabel(value) {
  if (missing(value)) return 'N/A';
  const number = Number(value);
  if (number > 0) return `${fixed(number, 2)}% better`;
  if (number < 0) return `${fixed(Math.abs(number), 2)}% worse`;
  return '0.00% change';
}

function overallComparisonCard(overall, experiment) {
  if (!overall || !experiment) return '';
  const paired = overall.paired_project_comparison || {};
  const ci = paired.improvement_95pct_ci || [];
  const verdict = overall.candidate_better ? 'Experiment currently beats production on cost MAE.' : 'Production currently beats the experiment on cost MAE.';
  return `<section class="panel">
    <div class="panel-head"><div><span class="kicker">Fresh same-cohort comparison</span><h2>Production vs latest experiment</h2></div></div>
    <div class="notice compact"><strong>${escape(experiment.experiment_name || experiment.experiment_id)}</strong> is an isolated cost-only candidate. It is compared with the actual freshly retrained production cost model and is never auto-promoted.</div>
    <div class="detail-financial">
      <div><span>Production cost MAE</span><strong>${fixed(overall.production_cost_mae)} pp</strong></div>
      <div><span>Experiment cost MAE</span><strong>${fixed(overall.experiment_cost_mae)} pp</strong></div>
      <div><span>Absolute MAE improvement</span><strong>${fixed(overall.absolute_mae_improvement_pp)} pp</strong></div>
      <div><span>Relative improvement</span><strong>${escape(improvementLabel(overall.improvement_percentage))}</strong></div>
      <div><span>Comparable test projects</span><strong>${missing(overall.comparison_test_projects) ? 'N/A' : overall.comparison_test_projects}</strong></div>
      <div><span>Comparable test snapshots</span><strong>${missing(overall.comparison_test_snapshots) ? 'N/A' : overall.comparison_test_snapshots}</strong></div>
      <div><span>Project-bootstrap chance experiment is better</span><strong>${missing(paired.probability_candidate_better) ? 'N/A' : `${fixed(Number(paired.probability_candidate_better) * 100, 1)}%`}</strong></div>
      <div><span>Bootstrap improvement 95% CI</span><strong>${ci.length === 2 ? `${fixed(ci[0], 2)}% to ${fixed(ci[1], 2)}%` : 'N/A'}</strong></div>
    </div>
    <div class="notice compact"><strong>Current evidence:</strong> ${escape(verdict)} Promotion remains a separate explicit decision.</div>
  </section>`;
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
  const comparison = prediction.comparison || null;
  const experiment = comparison?.experiment || null;
  const revealedComparison = actual?.comparison || null;
  const individualComparison = comparison ? `<section class="panel">
    <span class="kicker">Same project · before actual outcome reveal</span>
    <h2>Production vs latest experiment</h2>
    <div class="detail-financial">
      <div><span>Production predicted final cost overrun</span><strong>${fixed(comparison.production?.predicted_cost_overrun)}%</strong></div>
      <div><span>Experiment predicted final cost overrun</span><strong>${fixed(experiment?.predicted_cost_overrun)}%</strong></div>
      <div><span>Experiment current observed escalation</span><strong>${fixed(experiment?.current_observed_cost_escalation)}%</strong></div>
      <div><span>Experiment predicted remaining overrun</span><strong>${fixed(experiment?.predicted_remaining_cost_overrun)}%</strong></div>
      <div><span>Prediction difference</span><strong>${fixed(comparison.prediction_difference_pp)} pp</strong></div>
      <div><span>Experiment scope</span><strong>Cost only · delay/risk stay production</strong></div>
    </div>
    <div class="notice compact"><strong>Fair reveal:</strong> both predictions were generated before the official final outcome was sent to the browser.</div>
  </section>` : '';
  const individualReveal = revealedComparison ? `<section class="panel">
    <span class="kicker">Official outcome revealed once</span>
    <h2>Which model was closer for this project?</h2>
    <div class="detail-financial">
      <div><span>Actual final cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div>
      <div><span>Production prediction</span><strong>${fixed(revealedComparison.production_predicted_cost_overrun)}%</strong></div>
      <div><span>Experiment prediction</span><strong>${fixed(revealedComparison.experiment_predicted_cost_overrun)}%</strong></div>
      <div><span>Production absolute error</span><strong>${fixed(revealedComparison.production_cost_error_absolute_pp)} pp</strong></div>
      <div><span>Experiment absolute error</span><strong>${fixed(revealedComparison.experiment_cost_error_absolute_pp)} pp</strong></div>
      <div><span>Experiment error improvement</span><strong>${escape(improvementLabel(revealedComparison.individual_error_improvement_percentage))}</strong></div>
    </div>
    <div class="notice compact"><strong>Project verdict:</strong> ${revealedComparison.experiment_better_for_project ? 'The latest experiment was closer on cost for this held-out project.' : 'The production model was at least as close on cost for this held-out project.'}</div>
  </section>` : '';

  return `<div class="notice compact"><strong>Production run:</strong> ${escape(prediction.run_id || 'Not recorded')} · <strong>dataset:</strong> ${escape(shortFingerprint(prediction.dataset_fingerprint))}${experiment ? ` · <strong>experiment run:</strong> ${escape(experiment.experiment_run_id || 'Not recorded')}` : ''}</div>
  <div class="model-grid">
    <section class="panel">
      <span class="kicker">Production monthly lifecycle prediction generated first</span>
      <h2>${escape(text(prediction.project.project_name))}</h2>
      <div class="detail-financial">
        <div><span>Production predicted cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div>
        <div><span>Production predicted delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div>
        <div><span>Production predicted risk</span><strong>${escape(text(prediction.predicted_risk))} · ${fixed(prediction.risk_probability_percentage, 1)}%</strong></div>
        <div><span>Prediction snapshot</span><strong>${escape(text(prediction.snapshot_date, 'Unavailable'))}</strong></div>
        <div><span>Official history snapshots</span><strong>${missing(prediction.history_snapshots) ? 'N/A' : prediction.history_snapshots}</strong></div>
        <div><span>Confidence calibration</span><strong>${confidenceText}</strong></div>
        <div><span>Actual outcome sent yet?</span><strong>${prediction.audit.actual_outcomes_sent_to_browser ? 'Yes' : 'No'}</strong></div>
      </div>
      <h3>Lifecycle inputs visible to the production model</h3>
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
      <span class="kicker">Production explainability</span>
      <h2>Why the lifecycle model predicted this</h2>
      ${explanation}
      <div class="notice compact"><strong>Leakage audit:</strong> This project is excluded from the selected training years: ${prediction.audit.project_excluded_from_training ? 'YES' : 'NO'}.</div>
    </section>
  </div>
  ${individualComparison}
  ${actual ? `<section class="panel"><span class="kicker">Official outcome revealed after both predictions</span><h2>Production prediction vs actual</h2><div class="detail-financial"><div><span>Production cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div><div><span>Actual cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div><div><span>Production absolute cost error</span><strong>${fixed(actual.cost_error_absolute_pp)} pp</strong></div><div><span>Production delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div><div><span>Actual delay</span><strong>${fixed(actual.actual_delay_days)} days</strong></div><div><span>Absolute delay error</span><strong>${fixed(actual.delay_error_absolute_days)} days</strong></div><div><span>Production / actual risk</span><strong>${escape(text(prediction.predicted_risk))} / ${escape(text(actual.actual_risk))}</strong></div><div><span>Recorded completion</span><strong>${escape(text(actual.completion_date))}</strong></div></div><div class="notice compact"><strong>Reveal audit:</strong> ${escape(actual.reveal_policy)} · production run ${escape(actual.run_id || 'Not recorded')}</div>${source ? `<a class="secondary-btn" href="${escape(source)}" target="_blank" rel="noopener noreferrer">Open official PAIMANA source</a>` : ''}</section>${individualReveal}` : '<div class="notice compact"><strong>Actual outcome is still hidden.</strong> Reveal only after the judge has seen both production and experiment predictions.</div>'}`;
}

function trainingReceipt(registryRun, session = null, restored = false) {
  if (!registryRun) return 'No lifecycle model has been trained in this browser session yet.';
  const quality = registryRun.metrics?.metadata?.feature_quality || {};
  const baseline = registryRun.baseline_comparison || {};
  const algorithms = registryRun.selected_algorithms || {};
  const balanced = registryRun.balanced_stage_summary || registryRun.metrics?.metadata?.balanced_stage_summary || {};
  const sessionAudit = session ? ` <strong>Leakage guard:</strong> ${escape(session.leakage_guard)} Browser received actual held-out outcomes: <strong>${session.actual_outcomes_sent_to_browser ? 'YES' : 'NO'}</strong>.` : '';
  const provenance = registryRun.run_id ? ` <strong>Run ID:</strong> ${escape(registryRun.run_id)} · <strong>dataset:</strong> ${escape(shortFingerprint(registryRun.dataset_fingerprint))}.` : '';
  return `${restored ? '<strong>Restored active browser-session comparison.</strong> ' : ''}<strong>${escape(registryRun.model_version)} production retrained from scratch.</strong> Training: ${escape(registryRun.training_years)} · internal validation: ${escape(registryRun.internal_validation_year)} · untouched future holdout: ${escape(registryRun.testing_years)}. <strong>Lifecycle features:</strong> ${registryRun.feature_count} retained · ${quality.removed_invalid_feature_count ?? 'N/A'} rejected by the selected window audit. <strong>Selected production models:</strong> cost ${escape(algorithms.cost || 'unknown')} · delay ${escape(algorithms.delay || 'unknown')} · risk Random Forest. <strong>Fresh production holdout:</strong> cost MAE ${fixed(registryRun.metrics?.cost_model?.MAE)} pp · delay MAE ${fixed(registryRun.metrics?.delay_model?.MAE)} days · risk macro-F1 ${fractionPercent(registryRun.metrics?.risk_model?.macro_f1)}. <strong>Equal-stage diagnostic:</strong> cost MAE ${fixed(balanced.cost_mae)} pp · delay MAE ${fixed(balanced.delay_mae)} days · risk macro-F1 ${fractionPercent(balanced.risk_macro_f1)}. <strong>Five-feature benchmark:</strong> cost MAE ${fixed(baseline.cost_mae)} pp · delay MAE ${fixed(baseline.delay_mae)} days · risk macro-F1 ${fractionPercent(baseline.risk_macro_f1)}. <strong>Feature quality:</strong> ${fixed(quality.data_quality_score, 1)}%.${provenance}${sessionAudit}<br><a class="secondary-btn" href="#/prediction-accuracy">View Production Prediction Accuracy</a>`;
}

export async function ModelSimulationPage(root) {
  const catalog = await api.simulationVersions();
  if (!catalog.lifecycle_data_available) {
    root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1><p>Live comparison requires the official processed PAIMANA monthly lifecycle dataset.</p></div></header>
    <section class="panel"><div class="error-state">Official PAIMANA monthly lifecycle dataset is not available in this checkout.</div><p class="muted">Refresh the official archive separately, then rebuild <code>data/processed/paimana_monthly_snapshots.csv</code>. The comparison button does not download or parse PAIMANA PDFs.</p><button class="primary-btn" disabled>Retrain & Compare</button></section>`;
    return;
  }
  const years = catalog.data_years || [];
  if (!years.length) throw new Error('No identity-verified PAIMANA lifecycle years are available.');

  const latestExperiment = catalog.latest_experiment_id || 'exp_03';
  const savedRun = api.getActiveLifecycleRun();
  const yearNumbers = years.map((item) => item.year);
  const savedStart = Number(savedRun?.start_year);
  const savedEnd = Number(savedRun?.end_year);
  const defaultStart = yearNumbers.includes(savedStart) ? savedStart : yearNumbers[0];
  const preferredEnd = yearNumbers.filter((year) => year <= 2015).at(-1);
  const fallbackEnd = preferredEnd || yearNumbers[Math.max(0, Math.floor(yearNumbers.length / 2) - 1)];
  const defaultEnd = yearNumbers.includes(savedEnd) ? savedEnd : fallbackEnd;

  root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Production vs Experiment Verification</h1><p>Choose a historical training range. One click freshly retrains the production lifecycle stack and the latest isolated experiment on the same prepared PAIMANA evidence, then compares them overall and on the same judge-selected future project.</p></div></header>
  <div class="notice"><strong>Leakage rule:</strong> Projects completed after the training cutoff remain outside fitting. The experiment never replaces production automatically, and actual outcomes stay server-side until both project predictions have been generated.</div>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 1</span><h2>Retrain production and latest experiment</h2></div></div>
    <div class="filters">
      <label>Training start year<select id="custom-start">${yearOptions(years, defaultStart)}</select></label>
      <label>Training end year<select id="custom-end">${yearOptions(years, defaultEnd)}</select></label>
      <button class="primary-btn" id="custom-train">Retrain & Compare Production vs Latest Experiment</button>
    </div>
    <div id="training-receipt" class="notice compact">No production/experiment comparison has been trained in this browser session yet.</div>
    <div id="overall-comparison"></div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 2</span><h2>Judge chooses one unseen future project for both models</h2></div></div>
    <div class="filters">
      <label>Held-out completion year<select id="custom-test-year" disabled><option>Retrain & Compare first</option></select></label>
      <label>Official held-out project<select id="custom-project" disabled><option>Select a test year first</option></select></label>
      <button class="secondary-btn" id="random-project" disabled>Pick Random Unseen Project</button>
      <button class="primary-btn" id="custom-predict" disabled>Generate Both Predictions</button>
      <button class="secondary-btn" id="custom-reveal" disabled>Reveal Actual Outcome</button>
    </div>
    <div id="held-out-note" class="notice compact">After retraining, only future projects that both production and the latest experiment can score will be offered here.</div>
  </section>
  <div id="custom-output"></div>`;

  const start = root.querySelector('#custom-start');
  const end = root.querySelector('#custom-end');
  const trainButton = root.querySelector('#custom-train');
  const receipt = root.querySelector('#training-receipt');
  const overallNode = root.querySelector('#overall-comparison');
  const testYear = root.querySelector('#custom-test-year');
  const project = root.querySelector('#custom-project');
  const randomButton = root.querySelector('#random-project');
  const predictButton = root.querySelector('#custom-predict');
  const revealButton = root.querySelector('#custom-reveal');
  const heldOutNote = root.querySelector('#held-out-note');
  const output = root.querySelector('#custom-output');

  let session = null;
  let activeExperiment = null;
  let overallComparison = null;
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
    heldOutNote.innerHTML = '<div class="loading">Loading comparable held-out official lifecycle projects…</div>';
    try {
      const response = await api.comparisonProjects(session.comparison_session_id || session.session_id, Number(testYear.value));
      if (session.run_id && response.run_id !== session.run_id) throw new Error('Held-out project response belongs to a different production run. Retrain & Compare again.');
      if (session.dataset_fingerprint && response.dataset_fingerprint !== session.dataset_fingerprint) throw new Error('Held-out project response belongs to a different dataset snapshot. Retrain & Compare again.');
      if (session.experiment_run_id && response.experiment_run_id !== session.experiment_run_id) throw new Error('Held-out project response belongs to a different experiment run. Retrain & Compare again.');
      projectRows = response.items;
      project.innerHTML = projectRows.map((row) => `<option value="${row.record_index}">${escape(row.project_id)} · ${escape(row.project_name)}</option>`).join('');
      project.disabled = !projectRows.length;
      predictButton.disabled = !projectRows.length;
      randomButton.disabled = !projectRows.length;
      heldOutNote.innerHTML = `<strong>${projectRows.length} comparable held-out projects available for ${escape(response.year)}.</strong> ${escape(response.note)} No actual final cost, delay, completion date, or final expenditure has been sent to this page.`;
    } catch (error) {
      projectRows = [];
      project.innerHTML = '<option>No projects available</option>';
      heldOutNote.innerHTML = `<div class="error-state">${escape(error.message)}</div><p class="muted">If the backend restarted or either run was replaced, Retrain & Compare again.</p>`;
    }
  };

  const generatePrediction = async () => {
    if (!session || project.disabled) return;
    resetPrediction();
    predictButton.disabled = true;
    output.innerHTML = '<div class="loading">Generating production and latest-experiment predictions from the same held-out lifecycle snapshot…</div>';
    try {
      prediction = await api.predictComparison(session.comparison_session_id || session.session_id, Number(project.value));
      if (session.run_id && prediction.run_id !== session.run_id) throw new Error('Prediction belongs to a different production run. Retrain & Compare again.');
      if (session.dataset_fingerprint && prediction.dataset_fingerprint !== session.dataset_fingerprint) throw new Error('Prediction belongs to a different dataset snapshot. Retrain & Compare again.');
      if (session.experiment_run_id && prediction.comparison?.experiment?.experiment_run_id !== session.experiment_run_id) throw new Error('Prediction belongs to a different experiment run. Retrain & Compare again.');
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
    overallNode.innerHTML = '';
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
    receipt.innerHTML = '<div class="loading">Building one PAIMANA lifecycle dataset → retraining production → fitting the latest isolated experiment against that exact production feature/algorithm contract → evaluating both on one common future cohort → opening a judge comparison session…</div>';
    try {
      const comparisonRun = await api.retrainAndCompare(startYear, endYear, latestExperiment);
      const registryRun = comparisonRun.production;
      session = comparisonRun.session;
      activeExperiment = comparisonRun.experiment;
      overallComparison = comparisonRun.overall_comparison;
      if (registryRun.run_id && session.run_id !== registryRun.run_id) throw new Error('Comparison session did not bind to the exact retrained production run.');
      if (registryRun.dataset_fingerprint && session.dataset_fingerprint !== registryRun.dataset_fingerprint) throw new Error('Comparison session dataset fingerprint does not match the retrained production model.');
      if (activeExperiment.run_id && session.experiment_run_id !== activeExperiment.run_id) throw new Error('Comparison session did not bind to the exact experiment run.');

      const activeBase = {
        window: registryRun.window,
        model_version: registryRun.model_version,
        run_id: registryRun.run_id,
        dataset_fingerprint: registryRun.dataset_fingerprint,
        start_year: startYear,
        end_year: endYear,
        receipt: registryRun,
        session,
        experiment: activeExperiment,
        overall_comparison: overallComparison,
      };
      api.setActiveLifecycleRun(activeBase);
      receipt.innerHTML = trainingReceipt(registryRun, session);
      overallNode.innerHTML = overallComparisonCard(overallComparison, activeExperiment);

      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} comparable projects</option>`).join('');
      testYear.disabled = !eligible.length;
      if (eligible.length) await loadProjects();
    } catch (error) {
      session = null;
      activeExperiment = null;
      overallComparison = null;
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
      actual = await api.revealComparison(session.comparison_session_id || session.session_id, prediction.record_index);
      if (session.run_id && actual.run_id !== session.run_id) throw new Error('Reveal response belongs to a different production run.');
      if (session.dataset_fingerprint && actual.dataset_fingerprint !== session.dataset_fingerprint) throw new Error('Reveal response belongs to a different dataset snapshot.');
      if (session.experiment_run_id && actual.comparison?.experiment_run_id !== session.experiment_run_id) throw new Error('Reveal response belongs to a different experiment run.');
      output.innerHTML = predictionCard(prediction, actual);
    } catch (error) {
      output.innerHTML += `<div class="error-state">${escape(error.message)}</div>`;
      revealButton.disabled = false;
    }
  });

  if (savedRun?.receipt && savedRun.start_year === defaultStart && savedRun.end_year === defaultEnd) {
    receipt.innerHTML = trainingReceipt(savedRun.receipt, savedRun.session || null, true);
    activeExperiment = savedRun.experiment || null;
    overallComparison = savedRun.overall_comparison || null;
    overallNode.innerHTML = overallComparisonCard(overallComparison, activeExperiment);
    if (savedRun.session?.comparison_session_id) {
      session = savedRun.session;
      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} comparable projects</option>`).join('');
      testYear.disabled = !eligible.length;
      if (eligible.length) await loadProjects();
    } else {
      heldOutNote.innerHTML = '<strong>The production run remains selected for Prediction Accuracy.</strong> Retrain & Compare to create a fresh dual-model judge session.';
    }
  }
}
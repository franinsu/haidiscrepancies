const TYPE_LABELS = {
  arithmetic24: "24",
  maze: "Shortest path",
  grid_placement: "Grid placement",
  minesweeper_lite: "Minesweeper",
  mini_sudoku: "Mini Sudoku",
};

const TYPE_SHORT = {
  arithmetic24: "24",
  maze: "SP",
  grid_placement: "GP",
  minesweeper_lite: "MS",
  mini_sudoku: "SUD",
};

const TYPE_ORDER = ["arithmetic24", "maze", "grid_placement", "minesweeper_lite", "mini_sudoku"];

const MOVE_DELTAS = {
  U: [-1, 0],
  D: [1, 0],
  L: [0, -1],
  R: [0, 1],
};

const MOVE_BUTTONS = [
  ["U", "Up", "↑"],
  ["L", "Left", "←"],
  ["R", "Right", "→"],
  ["D", "Down", "↓"],
];

const SKIP_DELAY_MS = 90_000;
const TRIAL_TIMEOUT_MS = 120_000;
const BAD_RECORD_EXCLUSION_THRESHOLD = 5;
const MAX_INTERACTION_EVENTS_PER_TRIAL = 500;
const MAX_WRONG_ANSWER_ATTEMPTS_PER_TRIAL = 3;
const MAX_WRONG_ANSWER_ATTEMPT_EVENTS_STORED = 20;
const STUDY_VALID_WINDOW_MINUTES_BY_COHORT = {
  main: 100,
  module: 70,
};
const USERNAME_RE = /^[A-Za-z0-9_]+$/;
const ENTRY_PARAM_STORAGE_KEY = `reasoningStudyEntryParamsV1:${window.location.hostname}${window.location.pathname}`;
const SENSITIVE_QUERY_KEYS = [
  "PROLIFIC_PID",
  "STUDY_ID",
  "SESSION_ID",
  "access_token",
  "token",
  "participant",
  "prolific_pid",
  "prolific_study_id",
  "prolific_session_id",
  "prolific_token",
];

function readStoredEntryParams() {
  try {
    return new URLSearchParams(sessionStorage.getItem(ENTRY_PARAM_STORAGE_KEY) || "");
  } catch {
    return new URLSearchParams();
  }
}

function writeStoredEntryParams(params) {
  try {
    sessionStorage.setItem(ENTRY_PARAM_STORAGE_KEY, params.toString());
  } catch {
    // Formal registration still works in the current page load if sessionStorage is unavailable.
  }
}

function hasSensitiveQueryParams(params) {
  return SENSITIVE_QUERY_KEYS.some((key) => params.has(key));
}

function cleanVisibleQueryParams(params) {
  if (!window.history?.replaceState) return;
  const visible = new URLSearchParams(params);
  for (const key of SENSITIVE_QUERY_KEYS) {
    visible.delete(key);
  }
  const cleanQuery = visible.toString();
  const cleanUrl = `${window.location.pathname}${cleanQuery ? `?${cleanQuery}` : ""}${window.location.hash}`;
  window.history.replaceState(window.history.state, document.title, cleanUrl);
}

function shouldReuseStoredEntryParams(current) {
  if (hasSensitiveQueryParams(current)) return false;
  const [navigation] = performance?.getEntriesByType?.("navigation") || [];
  return navigation?.type === "reload" || navigation?.type === "back_forward";
}

function loadEntryParams() {
  const current = new URLSearchParams(window.location.search);
  const hasSensitive = hasSensitiveQueryParams(current);
  const merged = shouldReuseStoredEntryParams(current) ? readStoredEntryParams() : new URLSearchParams();
  for (const [key, value] of current.entries()) {
    merged.set(key, value);
  }
  if (hasSensitive) {
    writeStoredEntryParams(merged);
    cleanVisibleQueryParams(current);
  }
  return merged;
}

const URL_PARAMS = loadEntryParams();
const ACCESS_TOKEN = URL_PARAMS.get("access_token") || URL_PARAMS.get("token") || "";
const PROLIFIC_TOKEN = URL_PARAMS.get("prolific_token") || "";
const ADMIN_TEST_ASSIGNMENT_KEY = "reasoningStudyAdminTestAssignmentV1";
let PRACTICE_PUZZLES = [];
let ATTENTION_PUZZLES = [];
const state = {
  puzzles: [],
  plannedOrder: [],
  realOrder: [],
  practiceOrder: [],
  attentionOrder: [],
  order: [],
  index: 0,
  responseById: new Map(),
  initialResponses: new Map(),
  trialStartedAt: 0,
  trialStartedWallMs: 0,
  firstActionAt: 0,
  studyStartedAt: 0,
  studyStartedWallMs: 0,
  studyEndedAt: 0,
  timerId: null,
  participantId: "M0001",
  sessionId: "S01",
  blockId: "B01",
  currentUi: null,
  currentInteractionEvents: [],
  currentInteractionEventsTruncated: false,
  currentWrongAnswerAttempts: [],
  currentWrongAnswerAttemptsTruncated: false,
  datasetPath: "/data/stimuli/all_puzzles.jsonl",
  recordingResponse: false,
  awaitingNext: false,
  awaitingNextLabel: "",
  wrongSubmitAttempts: 0,
  breakStartedAt: 0,
  breakCount: 0,
  totalBreakTimeMs: 0,
  registered: false,
  backendMode: false,
  cohort: "main",
  studyValidWindowMinutes: STUDY_VALID_WINDOW_MINUTES_BY_COHORT.main,
  assignmentSource: "backend",
  testerMode: URL_PARAMS.get("tester") === "1" || URL_PARAMS.get("researcher_test") === "1",
  adminTestSession: URL_PARAMS.get("admin_test_session") === "1",
  formalMode: true,
  prolificPid: URL_PARAMS.get("PROLIFIC_PID") || URL_PARAMS.get("participant") || "",
  prolificStudyId: URL_PARAMS.get("STUDY_ID") || "",
  prolificSessionId: URL_PARAMS.get("SESSION_ID") || "",
  consentAt: "",
  payment: null,
  completionCode: "",
  participantSessionToken: "",
  testerLogin: "",
  assignmentVersion: "",
  assignmentHash: "",
  resuming: false,
  autoStartAdminTest: URL_PARAMS.get("autostart") === "1",
  assignmentMetaByPuzzle: new Map(),
  pendingJustAfterBreak: false,
  currentTrialAfterBreak: false,
};

const $ = (id) => document.getElementById(id);

function studyValidWindowMinutesForCohort(cohort) {
  return STUDY_VALID_WINDOW_MINUTES_BY_COHORT[cohort] || STUDY_VALID_WINDOW_MINUTES_BY_COHORT.main;
}

function currentStudyValidWindowMinutes() {
  return state.studyValidWindowMinutes || studyValidWindowMinutesForCohort(state.cohort);
}

function make(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function jsonClone(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return value;
  }
}

function trialElapsedMs() {
  return state.trialStartedAt ? Math.max(0, Math.round(performance.now() - state.trialStartedAt)) : null;
}

function studyElapsedMs() {
  return state.studyStartedAt ? Math.max(0, Math.round(performance.now() - state.studyStartedAt)) : null;
}

function safeCurrentUiAnswer() {
  try {
    return jsonClone(currentUiAnswer());
  } catch {
    return {};
  }
}

function recordInteraction(actionType, details = {}) {
  const puzzle = currentPuzzle();
  if (!state.studyStartedAt || !state.trialStartedAt || !puzzle || state.responseById.has(puzzle.id)) return;
  if (state.currentInteractionEvents.length >= MAX_INTERACTION_EVENTS_PER_TRIAL) {
    state.currentInteractionEventsTruncated = true;
    return;
  }
  state.currentInteractionEvents.push({
    event_index: state.currentInteractionEvents.length + 1,
    action_type: actionType,
    trial_elapsed_ms: trialElapsedMs(),
    study_elapsed_ms: studyElapsedMs(),
    display_trial_index: state.index + 1,
    puzzle_id: puzzle.id,
    puzzle_type: puzzle.puzzle_type,
    raw_answer_after: currentRawAnswer(),
    ui_answer_after: safeCurrentUiAnswer(),
    ...jsonClone(details),
  });
}

function describeClickTarget(target, event) {
  const rect = target.getBoundingClientRect ? target.getBoundingClientRect() : null;
  const ariaLabel = target.getAttribute("aria-label") || "";
  const rowColMatch = ariaLabel.match(/row\s+(\d+),\s*column\s+(\d+)/i);
  const description = {
    element_tag: target.tagName.toLowerCase(),
    element_id: target.id || "",
    element_classes: [...target.classList].slice(0, 8),
    text: (target.textContent || target.value || "").trim().replace(/\s+/g, " ").slice(0, 120),
    aria_label: ariaLabel,
    title: target.getAttribute("title") || "",
  };
  if (rowColMatch) {
    description.row = Number(rowColMatch[1]);
    description.column = Number(rowColMatch[2]);
  }
  if (rect) {
    description.click_offset_x = Math.round(event.clientX - rect.left);
    description.click_offset_y = Math.round(event.clientY - rect.top);
  }
  return description;
}

function recordAreaClick(event, surface) {
  const target = event.target instanceof Element ? event.target.closest("button, input, select, textarea") : null;
  if (!target) return;
  recordInteraction("ui_click", {
    surface,
    target: describeClickTarget(target, event),
  });
}

function recordWrongAnswerAttempt(validation = {}, source = "client") {
  state.wrongSubmitAttempts += 1;
  const attempt = {
    attempt_index: state.wrongSubmitAttempts,
    source,
    trial_elapsed_ms: trialElapsedMs(),
    study_elapsed_ms: studyElapsedMs(),
    display_trial_index: state.index + 1,
    raw_answer: currentRawAnswer().trim(),
    ui_answer: safeCurrentUiAnswer(),
    validation_message: validation.message || "",
    invalid_reason: validation.invalid_reason || validation.reason || validation.error || "",
  };
  if (state.currentWrongAnswerAttempts.length < MAX_WRONG_ANSWER_ATTEMPT_EVENTS_STORED) {
    state.currentWrongAnswerAttempts.push(attempt);
  } else {
    state.currentWrongAnswerAttemptsTruncated = true;
  }
  recordInteraction("wrong_answer_submit", {
    attempt_index: attempt.attempt_index,
    source,
    validation_message: attempt.validation_message,
    invalid_reason: attempt.invalid_reason,
  });
  return attempt;
}

function wrongAttemptFeedbackMessage(validation = {}) {
  const base = validation.message || "Incorrect. Try again.";
  const remaining = Math.max(0, MAX_WRONG_ANSWER_ATTEMPTS_PER_TRIAL - state.wrongSubmitAttempts);
  if (remaining === 0) {
    return `${base} The next incorrect submit will skip this trial.`;
  }
  const unit = remaining === 1 ? "attempt" : "attempts";
  return `${base} ${remaining} incorrect ${unit} left before this trial is skipped.`;
}

async function handleWrongAnswer(validation = {}, source = "client") {
  recordWrongAnswerAttempt(validation, source);
  const puzzle = currentPuzzle();
  if (isQualityPuzzle(puzzle)) {
    setStatus(`${validation.message || "Incorrect. Try again."} Practice and quick-check trials must be answered correctly before continuing.`, "warn");
    return;
  }
  if (state.wrongSubmitAttempts > MAX_WRONG_ANSWER_ATTEMPTS_PER_TRIAL) {
    await recordResponse({
      gaveUp: true,
      autoSkippedWrongLimit: true,
      validation: { ...validation, valid: false, complete: true },
    });
    return;
  }
  setStatus(wrongAttemptFeedbackMessage(validation), "warn");
}

function setStudyNavigationLocked(locked) {
  const brandLink = document.querySelector(".site-nav .brand-link");
  if (!brandLink) return;
  if (!brandLink.dataset.originalHref && brandLink.hasAttribute("href")) {
    brandLink.dataset.originalHref = brandLink.getAttribute("href");
  }
  if (locked) {
    brandLink.removeAttribute("href");
    brandLink.setAttribute("aria-disabled", "true");
    brandLink.setAttribute("tabindex", "-1");
    return;
  }
  if (brandLink.dataset.originalHref) {
    brandLink.setAttribute("href", brandLink.dataset.originalHref);
  }
  brandLink.removeAttribute("aria-disabled");
  brandLink.removeAttribute("tabindex");
}

function sanitizeUsernamePart(value) {
  const cleaned = String(value || "").replace(/[^A-Za-z0-9_]/g, "_").replace(/^_+|_+$/g, "");
  return cleaned || "participant";
}

function inferStudyFromHostname() {
  const host = window.location.hostname.toLowerCase();
  if (host.startsWith("main.")) return "main";
  if (host.startsWith("module.")) return "module";
  return "";
}

function configureFormalMode() {
  const requestedStudy = (URL_PARAMS.get("study") || URL_PARAMS.get("cohort") || inferStudyFromHostname()).toLowerCase();
  if (["main", "module"].includes(requestedStudy)) {
    state.cohort = requestedStudy;
  } else if (state.prolificPid && state.prolificPid.toUpperCase().startsWith("E")) {
    state.cohort = "module";
  }
  if (state.testerMode && !["main", "module"].includes(state.cohort)) {
    state.cohort = "main";
  }
  state.studyValidWindowMinutes = studyValidWindowMinutesForCohort(state.cohort);
  document.body.classList.add("formal-mode");
  if (state.testerMode) document.body.classList.add("tester-mode");
  $("formalInfo").classList.remove("hidden");
  $("testerLoginPanel").classList.toggle("hidden", !state.testerMode);
  const testerStudy = state.cohort === "module" ? "module" : "main";
  $("formalAdminTestLink").href = `../admin_test/?study=${testerStudy}`;
  const prefix =
    state.cohort === "module" ? "E" : "M";
  if (state.testerMode) {
    $("participantId").value = "Created after researcher login";
    $("participantId").readOnly = true;
    $("participantMeta").textContent = "Researcher test data are stored in the same database but excluded from Prolific quota and payment exports.";
    $("registerBtn").textContent = "Start researcher test session";
  } else {
    $("participantId").value = `${prefix}_${sanitizeUsernamePart(state.prolificPid || URL_PARAMS.get("participant") || "")}`;
    $("participantId").readOnly = Boolean(state.prolificPid);
    $("participantMeta").textContent = state.prolificPid
      ? `Prolific PID detected: ${state.prolificPid}`
      : "Enter your assigned username before registering.";
  }
  const formalCopy = {
    main: {
      name: "Human Reasoning - Main",
      pay: "You will first complete 3 short visual practice puzzles. The paid set has 100 short reasoning puzzles. Most puzzles are designed to be simple and direct; we estimate 60 minutes total. You will receive $12.00 base pay for a completed approved submission, plus $0.10 per correct paid puzzle, up to $22.00 total. Each paid trial has a 2-minute limit; skip unlocks after 90 seconds. If you submit more than 3 incorrect answers on one trial, that trial is automatically skipped and earns no bonus. To be counted as valid, you must finish within 100 minutes. The 100-minute clock starts when you begin the practice puzzles and ends when you submit the final puzzle.",
    },
    module: {
      name: "Human Reasoning - Modules",
      pay: "You will first complete 3 short visual practice puzzles. The paid set has 61-62 short reasoning puzzles. Most puzzles are designed to be simple and direct; we estimate 40 minutes total. You will receive $9.00 base pay for a completed approved submission, plus $0.10 per correct paid puzzle, up to about $15.20 total. Each paid trial has a 2-minute limit; skip unlocks after 90 seconds. If you submit more than 3 incorrect answers on one trial, that trial is automatically skipped and earns no bonus. To be counted as valid, you must finish within 70 minutes. The 70-minute clock starts when you begin the practice puzzles and ends when you submit the final puzzle.",
    },
  }[state.cohort];
  $("formalStudyName").textContent = state.testerMode ? `Researcher Test - ${formalCopy.name}` : formalCopy.name;
  $("formalPayCopy").textContent = state.testerMode
    ? `${formalCopy.pay} This researcher test path has no Prolific payment or bonus and does not count toward collection quota.`
    : formalCopy.pay;
  $("registerBtn").disabled = true;
}


async function loadQualityChecks() {
  const response = await fetch("/study_web/app/quality_checks.json");
  if (!response.ok) {
    throw new Error("Could not load practice and attention-check puzzles.");
  }
  const catalog = await response.json();
  PRACTICE_PUZZLES = Array.isArray(catalog.practice) ? catalog.practice : [];
  ATTENTION_PUZZLES = Array.isArray(catalog.attention) ? catalog.attention : [];
  if (PRACTICE_PUZZLES.length < 3 || ATTENTION_PUZZLES.length < 3) {
    throw new Error("Practice or attention-check catalog is incomplete.");
  }
}


function currentPuzzle() {
  return state.order[state.index];
}

function isPracticePuzzle(puzzle) {
  return Boolean(puzzle?.is_practice);
}

function isAttentionPuzzle(puzzle) {
  return Boolean(puzzle?.is_attention_check);
}

function isQualityPuzzle(puzzle) {
  return isPracticePuzzle(puzzle) || isAttentionPuzzle(puzzle);
}

function isModuleLikeCohort(cohort = state.cohort) {
  return cohort === "module";
}

function attentionCheckCountForCohort(cohort = state.cohort) {
  return isModuleLikeCohort(cohort) ? 2 : 3;
}

function realTrials() {
  return state.order.filter((puzzle) => !isQualityPuzzle(puzzle));
}

function realCompletedRows() {
  return [...state.responseById.values()].filter((row) => !row.is_practice && !row.is_attention_check);
}

function realTrialNumber(index = state.index) {
  return state.order.slice(0, index + 1).filter((puzzle) => !isQualityPuzzle(puzzle)).length;
}

function practiceTrialNumber(index = state.index) {
  return qualityTrialNumber(isPracticePuzzle, "is_practice", index);
}

function attentionTrialNumber(index = state.index) {
  return qualityTrialNumber(isAttentionPuzzle, "is_attention_check", index);
}

function rowPuzzleId(row, fallbackId = "") {
  return row?.puzzle_id || row?.check_id || fallbackId;
}

function qualityCompletedIds(rowFlag) {
  const ids = new Set();
  for (const [key, row] of state.responseById.entries()) {
    if (row?.[rowFlag]) ids.add(rowPuzzleId(row, key));
  }
  return ids;
}

function qualityTotal(predicate, rowFlag) {
  const ids = qualityCompletedIds(rowFlag);
  for (const puzzle of state.order) {
    if (predicate(puzzle)) ids.add(puzzle.id);
  }
  return ids.size;
}

function qualityTrialNumber(predicate, rowFlag, index = state.index) {
  const ids = qualityCompletedIds(rowFlag);
  for (const puzzle of state.order.slice(0, index + 1)) {
    if (predicate(puzzle)) ids.add(puzzle.id);
  }
  return ids.size;
}

function insertAttentionChecks(realOrder, attentionChecks) {
  if (!attentionChecks.length) return [...realOrder];
  const positions = [20, 40, 60].filter((position) => position <= realOrder.length).slice(0, attentionChecks.length);
  const out = [];
  realOrder.forEach((puzzle, index) => {
    out.push(puzzle);
    const paidCompleted = index + 1;
    const attentionIndex = positions.indexOf(paidCompleted);
    if (attentionIndex !== -1) out.push(attentionChecks[attentionIndex]);
  });
  return out;
}

function updatePlannedOrder() {
  if (!state.puzzles.length) return;
  if (!state.registered) {
    $("setupSummary").textContent = "Register a username to see the assigned puzzle IDs.";
    $("setupCounts").replaceChildren();
    $("setupPuzzleList").replaceChildren();
    $("registerBtn").disabled = state.formalMode && !$("consentCheck").checked;
    $("startBtn").classList.add("hidden");
    $("startBtn").disabled = true;
    return;
  }

  $("setupSummary").textContent = "";
  renderCounts($("setupCounts"), state.plannedOrder);
  renderPuzzleList($("setupPuzzleList"), state.plannedOrder, { setup: true });
  $("registerBtn").disabled = true;
  $("startBtn").classList.remove("hidden");
  $("startBtn").disabled = state.plannedOrder.length === 0;
}

function renderCounts(container, order) {
  const counts = countByType(order);
  container.replaceChildren(
    ...TYPE_ORDER.map((type) => {
      const chip = make("span", "count-chip");
      chip.textContent = `${TYPE_LABELS[type]}: ${counts[type] || 0}`;
      return chip;
    }),
  );
}

function countByType(order) {
  const counts = {};
  for (const puzzle of order) {
    counts[puzzle.puzzle_type] = (counts[puzzle.puzzle_type] || 0) + 1;
  }
  return counts;
}

function renderPuzzleList(container, order, options = {}) {
  const fragment = document.createDocumentFragment();
  order.forEach((puzzle, index) => {
    const item = make("div", "puzzle-list-item");
    const response = state.responseById.get(puzzle.id);
    const isCurrent = !options.setup && index === state.index;
    if (isCurrent) item.classList.add("current");
    if (response?.gave_up) item.classList.add("skipped");
    else if (response) item.classList.add("answered");
    else item.classList.add("pending");
    const moduleName = puzzle.module_metadata?.module;
    const conditionName = puzzle.module_metadata?.condition;
    const moduleLabel = moduleName && conditionName ? ` | ${moduleName}: ${conditionName}` : "";
    const practiceLabel = isPracticePuzzle(puzzle) ? " | practice" : isAttentionPuzzle(puzzle) ? " | attention check" : "";
    item.title = `${index + 1}. ${puzzle.id} | ${TYPE_LABELS[puzzle.puzzle_type]}${moduleLabel}${practiceLabel}`;

    const numberLabel = isPracticePuzzle(puzzle)
      ? `P${order.slice(0, index + 1).filter(isPracticePuzzle).length}`
      : isAttentionPuzzle(puzzle)
        ? `C${order.slice(0, index + 1).filter(isAttentionPuzzle).length}`
        : String(order.slice(0, index + 1).filter((item) => !isQualityPuzzle(item)).length).padStart(3, "0");
    const number = make("span", "list-number", numberLabel);
    const id = make("span", "list-id", puzzle.id);
    const type = make("span", `list-type ${puzzle.puzzle_type}`, TYPE_SHORT[puzzle.puzzle_type] || "?");
    item.append(number, id, type);
    fragment.append(item);
  });
  container.replaceChildren(fragment);
}

function startTrial() {
  const puzzle = currentPuzzle();
  state.currentUi = initUiState(puzzle);
  state.currentTrialAfterBreak = state.pendingJustAfterBreak;
  state.pendingJustAfterBreak = false;
  state.recordingResponse = false;
  state.awaitingNext = false;
  state.awaitingNextLabel = "";
  state.wrongSubmitAttempts = 0;
  state.currentInteractionEvents = [];
  state.currentInteractionEventsTruncated = false;
  state.currentWrongAnswerAttempts = [];
  state.currentWrongAnswerAttemptsTruncated = false;
  const practice = isPracticePuzzle(puzzle);
  const attention = isAttentionPuzzle(puzzle);
  const quality = practice || attention;
  const realTotal = realTrials().length;
  const practiceTotal = qualityTotal(isPracticePuzzle, "is_practice");
  const attentionTotal = qualityTotal(isAttentionPuzzle, "is_attention_check");
  $("progressText").textContent = practice
    ? `Practice ${practiceTrialNumber()} / ${practiceTotal}`
    : attention
      ? `Check ${attentionTrialNumber()} / ${attentionTotal}`
    : `Puzzle ${realTrialNumber()} / ${realTotal}`;
  $("metaText").textContent = practice
    ? `${TYPE_LABELS[puzzle.puzzle_type] || puzzle.puzzle_type} | practice, not paid`
    : attention
      ? `${TYPE_LABELS[puzzle.puzzle_type] || puzzle.puzzle_type} | attention check, not paid`
    : `${puzzle.id} | ${TYPE_LABELS[puzzle.puzzle_type] || puzzle.puzzle_type}`;
  $("promptTitle").textContent = practice
    ? `Practice: ${TYPE_LABELS[puzzle.puzzle_type] || "Puzzle"}`
    : attention
      ? `Quick Check: ${TYPE_LABELS[puzzle.puzzle_type] || "Puzzle"}`
      : TYPE_LABELS[puzzle.puzzle_type] || "Puzzle";
  renderPromptText(puzzle);
  $("visualArea").replaceChildren();
  $("answerArea").replaceChildren();
  $("statusText").textContent = "";
  $("statusText").className = "status";
  state.trialStartedAt = performance.now();
  state.trialStartedWallMs = Date.now();
  state.firstActionAt = 0;
  $("giveUpBtn").disabled = true;
  $("giveUpBtn").textContent = "I don't know";
  $("giveUpBtn").classList.toggle("hidden", isQualityPuzzle(puzzle));
  $("submitBtn").classList.remove("hidden");
  $("nextBtn").classList.add("hidden");
  $("nextBtn").disabled = true;
  $("breakBtn").classList.add("hidden");
  $("breakBtn").disabled = true;
  setInteractionLocked(false);

  if (puzzle.puzzle_type === "arithmetic24") {
    renderArithmetic(puzzle);
  } else if (puzzle.puzzle_type === "maze") {
    renderMaze(puzzle);
  } else if (puzzle.puzzle_type === "grid_placement") {
    renderGridPlacement(puzzle);
  } else if (puzzle.puzzle_type === "minesweeper_lite") {
    renderSingleCellGrid(puzzle, "mine");
  } else if (puzzle.puzzle_type === "mini_sudoku") {
    renderSingleCellGrid(puzzle, "sudoku");
  }
  refreshAnswer();
  renderStudyProgress();
  updateElapsedClock();
}

function promptForPuzzle(puzzle) {
  const inst = puzzle.machine_readable_instance;
  let prompt = "";
  if (puzzle.puzzle_type === "arithmetic24") {
    prompt = "Use each number exactly once with +, -, *, / and parentheses to make 24.";
    return withPresentationNotes(puzzle, prompt);
  }
  if (puzzle.puzzle_type === "maze") {
    prompt = "Draw one shortest path from S to G. Move only up, down, left, or right through open cells.";
    return withPresentationNotes(puzzle, prompt);
  }
  if (puzzle.puzzle_type === "grid_placement") {
    prompt = `Place ${inst.m} tokens. Tokens cannot be on X, and no two tokens can share a row or column.`;
    return withPresentationNotes(puzzle, prompt);
  }
  if (puzzle.puzzle_type === "minesweeper_lite") {
    const task = inst.target_kind === "safe" ? "safe" : "a mine";
    prompt = `Select exactly one hidden cell marked ?. Each number gives the exact count of mines in the up to eight neighboring cells, including diagonals. F marks a known mine. Your selected ? cell must be forced by the clues: in every mine arrangement consistent with all numbers, that cell is ${task}.`;
    return withPresentationNotes(puzzle, prompt);
  }
  if (puzzle.puzzle_type === "mini_sudoku") {
    prompt = `Select exactly one empty cell for digit ${inst.digit}. A cell is legal only if ${inst.digit} does not already appear in the same row, the same column, or the same outlined box. Printed numbers are fixed clues and cannot be selected.`;
    return withPresentationNotes(puzzle, prompt);
  }
  return withPresentationNotes(puzzle, puzzle.prompt_text);
}

function renderPromptText(puzzle) {
  const container = $("promptText");
  container.replaceChildren();
  const append = (text, bold = false) => {
    if (!text) return;
    container.append(bold ? make("strong", "prompt-keyword", text) : document.createTextNode(text));
  };
  const inst = puzzle.machine_readable_instance;
  if (puzzle.puzzle_type === "minesweeper_lite") {
    const target = inst.target_kind === "safe" ? "safe" : "mine";
    append(
      "Select exactly one hidden cell marked ?. Each number gives the exact count of mines in the up to eight neighboring cells, including diagonals. F marks a known mine. Your selected ? cell must be forced by the clues: in every mine arrangement consistent with all numbers, that cell is "
    );
    if (target === "mine") append("a ");
    append(target, true);
    append(".");
    appendPresentationNotes(container, puzzle);
    return;
  }
  if (puzzle.puzzle_type === "mini_sudoku") {
    const digit = String(inst.digit);
    append("Select exactly one empty cell for digit ");
    append(digit, true);
    append(". A cell is legal only if ");
    append(digit, true);
    append(" does not already appear in the same row, the same column, or the same outlined box. Printed numbers are fixed clues and cannot be selected.");
    appendPresentationNotes(container, puzzle);
    return;
  }
  append(promptForPuzzle(puzzle));
}

function appendPresentationNotes(container, puzzle) {
  const redundantRule = puzzle.presentation_metadata?.redundant_rule;
  if (redundantRule) container.append(document.createTextNode(`\n\n${redundantRule}`));
}

function withPresentationNotes(puzzle, prompt) {
  const redundantRule = puzzle.presentation_metadata?.redundant_rule;
  if (redundantRule) return `${prompt}\n\n${redundantRule}`;
  return prompt;
}

function initUiState(puzzle) {
  const inst = puzzle.machine_readable_instance;
  if (puzzle.puzzle_type === "arithmetic24") {
    return { tokens: [] };
  }
  if (puzzle.puzzle_type === "maze") {
    return { path: [findPoint(inst.grid, "S")] };
  }
  if (puzzle.puzzle_type === "grid_placement") {
    return { selected: [] };
  }
  if (puzzle.puzzle_type === "minesweeper_lite" || puzzle.puzzle_type === "mini_sudoku") {
    return { selectedCell: null };
  }
  return {};
}

function expressionFromTokens(tokens) {
  return tokens
    .map((token) => {
      if (token.type === "number") return String(token.value);
      if (token.value === "*") return " * ";
      if (token.value === "/") return " / ";
      if (token.value === "+") return " + ";
      if (token.value === "-") return " - ";
      return token.value;
    })
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

function renderArithmetic(puzzle) {
  setStatus("");
  const inst = puzzle.machine_readable_instance;
  const displayNumbers = inst.display_numbers || inst.numbers;
  const numbersPanel = make("div", "number-pad");
  const used = new Set(state.currentUi.tokens.filter((token) => token.type === "number").map((token) => token.index));

  displayNumbers.forEach((number, index) => {
    const button = make("button", "number-chip", String(number));
    button.type = "button";
    button.disabled = used.has(index);
    button.addEventListener("click", () => {
      state.currentUi.tokens.push({ type: "number", value: number, index });
      renderArithmetic(puzzle);
    });
    numbersPanel.append(button);
  });

  const opsPanel = make("div", "operator-pad");
  [
    ["+", "+"],
    ["-", "-"],
    ["*", "×"],
    ["/", "÷"],
    ["(", "("],
    [")", ")"],
  ].forEach(([value, label]) => {
    const button = make("button", "operator-button", label);
    button.type = "button";
    button.addEventListener("click", () => {
      state.currentUi.tokens.push({ type: "operator", value });
      renderArithmetic(puzzle);
    });
    opsPanel.append(button);
  });

  const editPanel = make("div", "edit-pad");
  const backspace = make("button", "secondary", "Backspace");
  backspace.type = "button";
  backspace.addEventListener("click", () => {
    state.currentUi.tokens.pop();
    renderArithmetic(puzzle);
  });
  const clear = make("button", "secondary", "Clear");
  clear.type = "button";
  clear.addEventListener("click", () => {
    state.currentUi.tokens = [];
    renderArithmetic(puzzle);
  });
  editPanel.append(backspace, clear);

  $("visualArea").replaceChildren(numbersPanel);
  $("answerArea").replaceChildren(opsPanel, editPanel);
  refreshAnswer();
}

function renderMaze(puzzle) {
  const inst = puzzle.machine_readable_instance;
  const grid = inst.grid;
  const rows = grid.length;
  const cols = grid[0].length;
  const path = state.currentUi.path;
  const pathIndex = new Map(path.map((coord, index) => [coordKey(coord), index]));
  const goal = findPoint(grid, "G");
  const cue = visualCueForPuzzle(puzzle);
  const cueKeys = new Set(
    (cue?.coordinates || []).map((coordinate) => coordKey([coordinate[0] - 1, coordinate[1] - 1]))
  );

  const board = make("div", "maze-grid");
  board.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const char = grid[r][c];
      const cell = make("button", "grid-cell maze-cell");
      const key = coordKey([r, c]);
      const index = pathIndex.get(key);
      cell.type = "button";
      cell.setAttribute("aria-label", `row ${r + 1}, column ${c + 1}`);
      if (char === "#") {
        cell.classList.add("wall");
        cell.disabled = true;
      } else {
        cell.addEventListener("click", () => addMazeCell(puzzle, r, c));
      }
      if (char === "S") {
        cell.textContent = "S";
        cell.classList.add("start");
      } else if (char === "G") {
        cell.textContent = "G";
        cell.classList.add("goal");
      } else if (index !== undefined) {
        cell.textContent = "•";
      }
      if (index !== undefined) {
        cell.classList.add("path");
        if (index === path.length - 1) cell.classList.add("current");
      }
      if (cueKeys.has(key)) {
        cell.classList.add("cue-cell");
        cell.title = "Highlighted cell";
      }
      board.append(cell);
    }
  }

  const movePad = make("div", "move-pad");
  MOVE_BUTTONS.forEach(([move, name, arrow]) => {
    const button = make("button", "move-button", arrow);
    button.type = "button";
    button.title = name;
    button.setAttribute("aria-label", name);
    button.addEventListener("click", () => moveMaze(puzzle, move));
    movePad.append(button);
  });

  const undo = make("button", "secondary", "Undo");
  undo.type = "button";
  undo.disabled = path.length <= 1;
  undo.addEventListener("click", () => {
    state.currentUi.path.pop();
    renderMaze(puzzle);
  });
  const reset = make("button", "secondary", "Reset path");
  reset.type = "button";
  reset.addEventListener("click", () => {
    state.currentUi.path = [findPoint(grid, "S")];
    renderMaze(puzzle);
  });

  const controls = make("div", "maze-controls");
  controls.append(movePad, undo, reset);
  if (cueKeys.size > 0) {
    const wrapper = make("div", "placement-wrap");
    wrapper.append(board);
    $("visualArea").replaceChildren(wrapper);
  } else {
    $("visualArea").replaceChildren(board);
  }
  $("answerArea").replaceChildren(controls);

  if (sameCoord(last(path), goal)) {
    setStatus("Path complete. Submit when finished.");
  } else {
    setStatus("");
  }
  refreshAnswer();
}

function addMazeCell(puzzle, r, c) {
  const inst = puzzle.machine_readable_instance;
  const grid = inst.grid;
  const path = state.currentUi.path;
  const cell = [r, c];
  const current = last(path);
  const previous = path[path.length - 2];

  if (!inBounds(grid, r, c)) {
    setStatus("That move leaves the board.", "warn");
    return;
  }
  if (previous && sameCoord(previous, cell)) {
    path.pop();
    renderMaze(puzzle);
    return;
  }
  if (sameCoord(current, findPoint(grid, "G"))) {
    setStatus("You are already at G. Submit or undo.", "warn");
    return;
  }
  if (!isAdjacent(current, cell)) {
    setStatus("Choose a neighboring open square.", "warn");
    return;
  }
  if (grid[r][c] === "#") {
    setStatus("That square is blocked.", "warn");
    return;
  }
  if (path.some((coord) => sameCoord(coord, cell))) {
    setStatus("This interface uses simple paths, so a square cannot be reused.", "warn");
    return;
  }
  path.push(cell);
  renderMaze(puzzle);
}

function moveMaze(puzzle, move) {
  const [dr, dc] = MOVE_DELTAS[move];
  const [r, c] = last(state.currentUi.path);
  addMazeCell(puzzle, r + dr, c + dc);
}

function renderGridPlacement(puzzle) {
  setStatus("");
  const inst = puzzle.machine_readable_instance;
  const { board, m, n } = inst;
  const selected = state.currentUi.selected;
  const selectedKeys = new Set(selected.map(coordKey));
  const cue = visualCueForPuzzle(puzzle);
  const cueKeys = new Set(
    (cue?.coordinates || []).map((coordinate) => coordKey([coordinate[0] - 1, coordinate[1] - 1]))
  );

  const wrapper = make("div", "placement-wrap");
  const grid = make("div", "placement-grid");
  grid.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;

  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const cell = make("button", "grid-cell placement-cell");
      const key = coordKey([r, c]);
      cell.type = "button";
      cell.setAttribute("aria-label", `row ${r + 1}, column ${c + 1}`);
      if (board[r][c] === "X") {
        cell.textContent = "X";
        cell.disabled = true;
        cell.classList.add("blocked");
      } else {
        cell.addEventListener("click", () => togglePlacement(puzzle, r, c));
        if (selectedKeys.has(key)) {
          cell.textContent = "O";
          cell.classList.add("token");
        }
      }
      if (cueKeys.has(key)) {
        cell.classList.add("cue-cell");
        cell.title = "Highlighted cell";
      }
      grid.append(cell);
    }
  }

  const legend = make("div", "placement-legend");
  legend.innerHTML = `<span><b>${selected.length}</b> / ${m} tokens placed</span>`;
  if (cueKeys.size > 0) {
    wrapper.append(grid, legend);
  } else {
    wrapper.append(grid, legend);
  }

  const editPanel = make("div", "edit-pad");
  const clear = make("button", "secondary", "Clear tokens");
  clear.type = "button";
  clear.addEventListener("click", () => {
    state.currentUi.selected = [];
    renderGridPlacement(puzzle);
  });
  editPanel.append(clear);

  $("visualArea").replaceChildren(wrapper);
  $("answerArea").replaceChildren(editPanel);
  refreshAnswer();
}

function togglePlacement(puzzle, r, c) {
  const { board, m } = puzzle.machine_readable_instance;
  const selected = state.currentUi.selected;
  const index = selected.findIndex((coord) => sameCoord(coord, [r, c]));
  if (index >= 0) {
    selected.splice(index, 1);
    renderGridPlacement(puzzle);
    return;
  }
  if (board[r][c] === "X") {
    setStatus("Tokens cannot be placed on X.", "warn");
    return;
  }
  if (selected.length >= m) {
    setStatus(`You already placed ${m} tokens. Remove one to change it.`, "warn");
    return;
  }
  if (selected.some(([sr]) => sr === r)) {
    setStatus("That row already has a token.", "warn");
    return;
  }
  if (selected.some(([, sc]) => sc === c)) {
    setStatus("That column already has a token.", "warn");
    return;
  }
  selected.push([r, c]);
  renderGridPlacement(puzzle);
}

function renderSingleCellGrid(puzzle, mode) {
  setStatus("");
  const inst = puzzle.machine_readable_instance;
  const board = inst.board;
  const rows = board.length;
  const cols = board[0].length;
  const boxRows = Number(inst.box_rows || inst.boxRows || 0);
  const boxCols = Number(inst.box_cols || inst.boxCols || 0);
  const selected = state.currentUi.selectedCell;
  const cue = visualCueForPuzzle(puzzle);
  const cueKeys = new Set(
    (cue?.coordinates || []).map((coordinate) => coordKey([coordinate[0] - 1, coordinate[1] - 1]))
  );
  const wrapper = make("div", "placement-wrap");
  const grid = make("div", `${mode}-grid single-cell-grid`);
  grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  if (mode === "sudoku") {
    grid.style.setProperty("--sudoku-cols", String(cols));
    grid.style.setProperty("--sudoku-rows", String(rows));
  }

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const ch = board[r][c];
      const cell = make("button", `grid-cell ${mode}-cell`);
      cell.type = "button";
      cell.setAttribute("aria-label", `row ${r + 1}, column ${c + 1}`);
      const open = mode === "mine" ? ch === "?" : ch === ".";
      if (open) {
        cell.textContent = selected && selected[0] === r && selected[1] === c ? "✓" : mode === "mine" ? "?" : "";
        cell.addEventListener("click", () => {
          state.currentUi.selectedCell = selected && selected[0] === r && selected[1] === c ? null : [r, c];
          renderSingleCellGrid(puzzle, mode);
        });
      } else {
        cell.textContent = ch;
        cell.disabled = true;
        cell.classList.add(mode === "mine" ? "clue" : "fixed");
        if (mode === "mine" && ch === "F") cell.classList.add("known-mine");
      }
      if (mode === "sudoku") {
        const boxR = boxRows ? Math.floor(r / boxRows) : 0;
        const boxC = boxCols ? Math.floor(c / boxCols) : 0;
        cell.classList.add((boxR + boxC) % 2 === 0 ? "sudoku-box-even" : "sudoku-box-odd");
        if (boxRows && r % boxRows === 0) cell.classList.add("box-top");
        if (boxRows && (r + 1) % boxRows === 0) cell.classList.add("box-bottom");
        if (boxCols && c % boxCols === 0) cell.classList.add("box-left");
        if (boxCols && (c + 1) % boxCols === 0) cell.classList.add("box-right");
      }
      if (selected && selected[0] === r && selected[1] === c) {
        cell.classList.add("chosen-cell");
      }
      if (cueKeys.has(coordKey([r, c]))) {
        cell.classList.add("cue-cell");
        cell.title = "Highlighted cell";
      }
      grid.append(cell);
    }
  }
  wrapper.append(grid);
  $("visualArea").replaceChildren(wrapper);
  $("answerArea").replaceChildren();
  refreshAnswer();
}

function visualCueForPuzzle(puzzle) {
  const instCue = puzzle.machine_readable_instance?.visual_cue || puzzle.machine_readable_instance?.irrelevant_cue;
  const presentationCue = puzzle.presentation_metadata?.visual_cue || puzzle.presentation_metadata?.irrelevant_cue;
  const cue = instCue || presentationCue;
  if (!cue) return null;
  if (Array.isArray(cue.coordinates) && cue.coordinates.length > 0) {
    return { ...cue, coordinates: cue.coordinates };
  }
  if (Array.isArray(cue.coordinate) && cue.coordinate.length === 2) {
    return { ...cue, coordinates: [cue.coordinate] };
  }
  return null;
}

function refreshAnswer() {
  const puzzle = currentPuzzle();
  if (state.awaitingNext) return;
  const raw = currentRawAnswer();
  const complete = hasCompleteAnswer(puzzle);
  $("answerPreview").textContent = raw || "No answer yet";
  $("answerMeta").textContent = answerMeta(puzzle);
  $("submitBtn").disabled = !complete;
  if (complete) setStatus("Answer ready. Submit when finished.");
}

function currentRawAnswer() {
  const puzzle = currentPuzzle();
  if (!puzzle || !state.currentUi) return "";
  if (puzzle.puzzle_type === "arithmetic24") {
    return expressionFromTokens(state.currentUi.tokens);
  }
  if (puzzle.puzzle_type === "maze") {
    return movesFromCoords(state.currentUi.path);
  }
  if (puzzle.puzzle_type === "grid_placement") {
    return coordinatesAnswer(state.currentUi.selected);
  }
  if (puzzle.puzzle_type === "minesweeper_lite" || puzzle.puzzle_type === "mini_sudoku") {
    return state.currentUi.selectedCell ? coordinatesAnswer([state.currentUi.selectedCell]) : "";
  }
  return "";
}

function currentUiAnswer() {
  const puzzle = currentPuzzle();
  if (puzzle.puzzle_type === "arithmetic24") {
    return { expression: currentRawAnswer() };
  }
  if (puzzle.puzzle_type === "maze") {
    return {
      moves: currentRawAnswer(),
      coordinates: state.currentUi.path.map(toOneBased),
      length: Math.max(0, state.currentUi.path.length - 1),
    };
  }
  if (puzzle.puzzle_type === "grid_placement") {
    return { coordinates: sortedOneBased(state.currentUi.selected) };
  }
  if (puzzle.puzzle_type === "minesweeper_lite" || puzzle.puzzle_type === "mini_sudoku") {
    return { coordinate: state.currentUi.selectedCell ? toOneBased(state.currentUi.selectedCell) : null };
  }
  return {};
}

function answerMeta(puzzle) {
  if (!puzzle) return "";
  if (puzzle.puzzle_type === "arithmetic24") {
    const used = state.currentUi.tokens.filter((token) => token.type === "number").length;
    return `${used} / 4 numbers used`;
  }
  if (puzzle.puzzle_type === "maze") {
    return `${Math.max(0, state.currentUi.path.length - 1)} moves`;
  }
  if (puzzle.puzzle_type === "grid_placement") {
    return `${state.currentUi.selected.length} / ${puzzle.machine_readable_instance.m} tokens`;
  }
  if (puzzle.puzzle_type === "minesweeper_lite" || puzzle.puzzle_type === "mini_sudoku") {
    return state.currentUi.selectedCell ? "1 selected" : "0 selected";
  }
  return "";
}

function hasCompleteAnswer(puzzle) {
  if (puzzle.puzzle_type === "arithmetic24") {
    const numberTokens = state.currentUi.tokens.filter((token) => token.type === "number");
    const hasOperator = state.currentUi.tokens.some((token) => token.type === "operator" && "+-*/".includes(token.value));
    return numberTokens.length === 4 && hasOperator;
  }
  if (puzzle.puzzle_type === "maze") {
    return sameCoord(last(state.currentUi.path), findPoint(puzzle.machine_readable_instance.grid, "G"));
  }
  if (puzzle.puzzle_type === "grid_placement") {
    return state.currentUi.selected.length === puzzle.machine_readable_instance.m;
  }
  if (puzzle.puzzle_type === "minesweeper_lite" || puzzle.puzzle_type === "mini_sudoku") {
    return Boolean(state.currentUi.selectedCell);
  }
  return Boolean(currentRawAnswer());
}

function responseOutcome(row) {
  if (!row) return "";
  if (row.timeout) return "timeout";
  if (row.gave_up || row.skipped) return "skip";
  if (row.is_correct) return "correct";
  return "recorded";
}

function previousResponseContext() {
  let previousDisplayed = null;
  let previousPaid = null;
  for (let index = state.index - 1; index >= 0; index -= 1) {
    const puzzle = state.order[index];
    const row = state.responseById.get(puzzle.id);
    if (!row) continue;
    if (!previousDisplayed) previousDisplayed = { puzzle, row, displayIndex: index + 1 };
    if (!isQualityPuzzle(puzzle)) {
      previousPaid = { puzzle, row, displayIndex: index + 1 };
      break;
    }
  }
  return { previousDisplayed, previousPaid };
}

function moduleConditionOccurrence(puzzle) {
  const module = puzzle.module_metadata?.module || "main";
  const condition = puzzle.module_metadata?.condition || puzzle.variant_type || "canonical";
  let count = 0;
  for (let index = 0; index <= state.index; index += 1) {
    const candidate = state.order[index];
    if (isQualityPuzzle(candidate)) continue;
    const candidateModule = candidate.module_metadata?.module || "main";
    const candidateCondition = candidate.module_metadata?.condition || candidate.variant_type || "canonical";
    if (candidateModule === module && candidateCondition === condition) count += 1;
  }
  return count;
}

function assignmentMetaForPuzzle(puzzle) {
  return state.assignmentMetaByPuzzle.get(puzzle.id) || {};
}

function buildAssignmentMeta(assignment) {
  const meta = new Map();
  (assignment.unit_plan || []).forEach((choice, unitIndex) => {
    (choice.trial_ids || []).forEach((trialId, trialIndex) => {
      meta.set(trialId, {
        unit_index: unitIndex + 1,
        unit_id: choice.unit_id || "",
        choice_id: choice.choice_id || choice.sequence_id || trialId,
        trial_index_in_unit: trialIndex + 1,
        module: choice.module || "",
        condition: choice.condition || "",
        sequence_id: choice.sequence_id || "",
      });
    });
  });
  return meta;
}

async function submitCurrentAnswer() {
  if (state.awaitingNext) return;
  const puzzle = currentPuzzle();
  if (!puzzle || state.responseById.has(puzzle.id)) return;
  if (!hasCompleteAnswer(puzzle)) {
    setStatus("Complete an answer before submitting.", "warn");
    return;
  }
  if (state.backendMode) {
    await recordResponse({ correct: true, validation: { valid: true, complete: true }, serverScored: true });
    return;
  }
  const validation = validateCurrentAnswer(puzzle);
  if (!validation.valid) {
    await handleWrongAnswer(validation, "client");
    return;
  }
  await recordResponse({ correct: true, validation });
}

async function recordResponse(outcome = {}) {
  if (state.recordingResponse) return;
  const puzzle = currentPuzzle();
  if (!puzzle || state.responseById.has(puzzle.id)) return;
  const practice = isPracticePuzzle(puzzle);
  const attention = isAttentionPuzzle(puzzle);
  const quality = practice || attention;
  const now = performance.now();
  const elapsedMs = Math.max(0, Math.round(now - state.trialStartedAt));
  const hardTimedOut = !quality && elapsedMs >= TRIAL_TIMEOUT_MS;
  const timedOut = Boolean(outcome.timeout) || hardTimedOut;
  const gaveUp = Boolean(outcome.gaveUp) && !timedOut;
  const autoSkippedWrongLimit = Boolean(outcome.autoSkippedWrongLimit);
  if (gaveUp && !timedOut && !autoSkippedWrongLimit && now - state.trialStartedAt < SKIP_DELAY_MS) {
    setStatus("I don't know is available after 90 seconds.", "warn");
    return;
  }
  const validation = outcome.validation || validateCurrentAnswer(puzzle);
  const correct = Boolean(outcome.correct);
  if (!gaveUp && !timedOut && !correct && !hasCompleteAnswer(puzzle)) {
    setStatus("Complete an answer before submitting.", "warn");
    return;
  }
  state.recordingResponse = true;
  const raw = timedOut ? "timeout" : autoSkippedWrongLimit ? "wrong attempt limit exceeded" : gaveUp ? "I don't know" : currentRawAnswer().trim();
  const trialEndWallMs = Date.now();
  let solutionId = validation.valid ? validation.solution?.solution_id || null : null;
  let abstractSolutionId = validation.valid ? validation.solution?.abstract_solution_id || null : null;
  let isCorrect = !gaveUp && !timedOut && correct && validation.valid;
  const badRecord = Boolean(gaveUp || timedOut);
  if (autoSkippedWrongLimit) {
    recordInteraction("wrong_attempt_limit_recorded", {
      outcome: "wrong_attempt_limit",
      wrong_attempt_limit: MAX_WRONG_ANSWER_ATTEMPTS_PER_TRIAL,
    });
  } else if (gaveUp) {
    recordInteraction("give_up_recorded", { outcome: "gave_up" });
  } else if (timedOut) {
    recordInteraction("timeout_recorded", { outcome: "timeout" });
  }
  const { previousDisplayed, previousPaid } = previousResponseContext();
  const assignmentMeta = assignmentMetaForPuzzle(puzzle);
  const tooFast = !quality && isCorrect && Math.round(now - state.trialStartedAt) > 0 && Math.round(now - state.trialStartedAt) <= 2000;
  const testerData = state.testerMode || state.adminTestSession;
  const responseRow = {
    source: testerData ? "researcher_tester" : "human",
    participant_id: state.participantId,
    username: state.participantId,
    participant_source: testerData ? "tester" : state.prolificPid ? "prolific" : state.backendMode ? "backend_local" : "local",
    tester_login: state.testerLogin || "",
    platform_id: state.prolificPid || "",
    prolific_pid: state.prolificPid || "",
    prolific_study_id: state.prolificStudyId || "",
    prolific_session_id: state.prolificSessionId || "",
    session_id: state.sessionId,
    block_id: state.blockId,
    cohort: state.cohort,
    assignment_version: state.assignmentVersion || "",
    assignment_hash: state.assignmentHash || "",
    participant_session_token: state.participantSessionToken || "",
    trial_index: quality ? null : realTrialNumber(),
    paid_trial_position: quality ? null : realTrialNumber(),
    display_trial_index: state.index + 1,
    display_trial_count: state.order.length,
    assigned_trial_count: realTrials().length,
    assigned_trial_ids: realTrials().map((trial) => trial.id),
    puzzle_id: puzzle.id,
    family_id: puzzle.family_id || puzzle.base_puzzle_id || null,
    module: puzzle.module_metadata?.module || "main",
    condition: puzzle.module_metadata?.condition || puzzle.variant_type || "canonical",
    condition_occurrence_index: quality ? null : moduleConditionOccurrence(puzzle),
    module_unit_index: assignmentMeta.unit_index || null,
    module_trial_index_in_unit: assignmentMeta.trial_index_in_unit || null,
    module_choice_id: assignmentMeta.choice_id || null,
    module_unit_id: assignmentMeta.unit_id || null,
    sequence_id: puzzle.module_metadata?.sequence_id || null,
    sequence_position: puzzle.module_metadata?.sequence_position || null,
    puzzle_type: puzzle.puzzle_type,
    raw_answer: raw,
    raw_response: raw,
    parsed_answer: raw,
    study_start_time_iso: new Date(state.studyStartedWallMs).toISOString(),
    trial_start_time_iso: new Date(state.trialStartedWallMs).toISOString(),
    trial_end_time_iso: new Date(trialEndWallMs).toISOString(),
    response_time_ms: timedOut ? TRIAL_TIMEOUT_MS : elapsedMs,
    first_action_time_ms: state.firstActionAt ? Math.round(state.firstActionAt - state.trialStartedAt) : null,
    timeout: timedOut,
    gave_up: Boolean(gaveUp),
    skipped: Boolean(gaveUp),
    is_correct: isCorrect,
    wrong_submit_attempts: state.wrongSubmitAttempts,
    wrong_submit_attempts_before_correct: isCorrect ? state.wrongSubmitAttempts : null,
    wrong_answer_attempts: jsonClone(state.currentWrongAnswerAttempts),
    wrong_answer_attempt_count: state.wrongSubmitAttempts,
    wrong_answer_attempts_truncated: state.currentWrongAnswerAttemptsTruncated,
    wrong_attempt_limit: MAX_WRONG_ANSWER_ATTEMPTS_PER_TRIAL,
    auto_skipped_wrong_limit: autoSkippedWrongLimit,
    bad_record: badRecord,
    is_practice: practice,
    is_attention_check: attention,
    quota_counted: !quality,
    bonus_earned: !quality && isCorrect,
    bonus_usd: isCorrect ? state.payment?.correct_bonus_per_trial_usd || 0 : 0,
    too_fast_response_flag: tooFast,
    just_after_break: Boolean(state.currentTrialAfterBreak),
    previous_displayed_puzzle_id: previousDisplayed?.puzzle.id || null,
    previous_displayed_puzzle_type: previousDisplayed?.puzzle.puzzle_type || null,
    previous_displayed_was_practice: previousDisplayed ? isPracticePuzzle(previousDisplayed.puzzle) : null,
    previous_displayed_was_attention_check: previousDisplayed ? isAttentionPuzzle(previousDisplayed.puzzle) : null,
    previous_displayed_outcome: previousDisplayed ? responseOutcome(previousDisplayed.row) : null,
    previous_displayed_response_time_ms: previousDisplayed?.row.response_time_ms || null,
    previous_paid_puzzle_id: previousPaid?.puzzle.id || null,
    previous_paid_puzzle_type: previousPaid?.puzzle.puzzle_type || null,
    previous_paid_outcome: previousPaid ? responseOutcome(previousPaid.row) : null,
    previous_paid_response_time_ms: previousPaid?.row.response_time_ms || null,
    client_scored: true,
    solution_id: isCorrect ? solutionId : null,
    abstract_solution_id: isCorrect ? abstractSolutionId : null,
    accepted_solution_id: isCorrect ? solutionId : null,
    accepted_abstract_solution_id: isCorrect ? abstractSolutionId : null,
    notes: "",
    ui_answer: gaveUp || timedOut ? {} : currentUiAnswer(),
    break_count_so_far: state.breakCount,
    total_break_time_ms_so_far: Math.round(state.totalBreakTimeMs),
    interaction_events: jsonClone(state.currentInteractionEvents),
    interaction_event_count: state.currentInteractionEvents.length,
    interaction_events_truncated: state.currentInteractionEventsTruncated,
    collection_interface: "visual_click_final_submit_v5",
  };
  if (practice) {
    responseRow.bonus_usd = 0;
  }
  if (quality) {
    responseRow.bonus_usd = 0;
  }
  if (state.backendMode) {
    try {
      setStatus("Saving answer...", "");
      $("submitBtn").disabled = true;
      $("giveUpBtn").disabled = true;
      const saved = quality ? await sendQualityCheckToBackend(responseRow) : await sendResponseToBackend(responseRow);
      if (saved?.payment) state.payment = saved.payment;
      if (saved?.is_correct) {
        solutionId = saved.solution_id || solutionId;
        abstractSolutionId = saved.abstract_solution_id || abstractSolutionId || solutionId;
        isCorrect = true;
        Object.assign(responseRow, {
          is_correct: true,
          server_scored: true,
          solution_id: solutionId,
          abstract_solution_id: abstractSolutionId,
          accepted_solution_id: saved.accepted_solution_id || solutionId,
          accepted_abstract_solution_id: saved.accepted_abstract_solution_id || abstractSolutionId,
          parsed_answer: saved.parsed_answer ?? responseRow.parsed_answer,
          canonical_answer: saved.canonical_answer ?? responseRow.canonical_answer,
          parse_confidence: saved.parse_confidence ?? responseRow.parse_confidence,
          requires_manual_review: Boolean(saved.requires_manual_review),
        });
      }
    } catch (error) {
      state.recordingResponse = false;
      setInteractionLocked(false);
      $("submitBtn").disabled = !hasCompleteAnswer(puzzle);
      $("giveUpBtn").disabled = performance.now() - state.trialStartedAt < SKIP_DELAY_MS;
      const errorMessage = String(error.message || "");
      if (errorMessage.startsWith("invalid_final_answer:") || errorMessage.startsWith("invalid_quality_check_answer:")) {
        await handleWrongAnswer(
          {
            message: "Incorrect. Try again.",
            invalid_reason: errorMessage.split(":").slice(1).join(":") || errorMessage,
          },
          "server",
        );
      } else {
        setStatus(`Could not save answer: ${error.message || "backend error"}. Please retry.`, "warn");
      }
      return;
    }
  }
  state.responseById.set(puzzle.id, responseRow);

  showRecordedTrialState({ isCorrect, gaveUp, timedOut, autoSkippedWrongLimit });
}

function showRecordedTrialState({ isCorrect, gaveUp, timedOut, autoSkippedWrongLimit = false }) {
  state.awaitingNext = true;
  state.awaitingNextLabel = isCorrect ? "Correct" : timedOut ? "Timeout" : autoSkippedWrongLimit ? "Skipped" : "Skipped";
  setInteractionLocked(true);
  $("submitBtn").classList.add("hidden");
  $("giveUpBtn").classList.add("hidden");
  $("nextBtn").classList.remove("hidden");
  $("nextBtn").disabled = false;
  $("nextBtn").textContent = hasRemainingTrials() ? "Next puzzle" : "Finish";
  $("breakBtn").classList.remove("hidden");
  $("breakBtn").disabled = false;
  if (isCorrect) setStatus("Correct.", "ok");
  else if (timedOut) setStatus("Time is up. This trial was recorded.", "warn");
  else if (autoSkippedWrongLimit) setStatus("Incorrect answer limit reached. This trial was skipped and recorded.", "warn");
  else if (gaveUp) setStatus("Skipped. This trial was recorded.", "warn");
  renderStudyProgress();
}

function setInteractionLocked(locked) {
  ["visualArea", "answerArea"].forEach((id) => {
    $(id).querySelectorAll("button").forEach((button) => {
      button.disabled = locked || button.disabled;
    });
  });
}

function hasRemainingTrials() {
  return state.order.some((candidate, index) => index > state.index && !state.responseById.has(candidate.id));
}

function goToNextAfterCorrect() {
  if (!state.awaitingNext) return;
  const nextIndex = state.order.findIndex((candidate, index) => index > state.index && !state.responseById.has(candidate.id));
  if (nextIndex === -1) {
    finish();
  } else {
    state.index = nextIndex;
    startTrial();
  }
}

function renderStudyProgress() {
  const total = realTrials().length;
  const rows = realCompletedRows();
  const practiceDone = qualityCompletedIds("is_practice").size;
  const attentionDone = qualityCompletedIds("is_attention_check").size;
  const practiceTotal = qualityTotal(isPracticePuzzle, "is_practice");
  const attentionTotal = qualityTotal(isAttentionPuzzle, "is_attention_check");
  const complete = rows.length;
  const remaining = Math.max(0, total - complete);
  const correct = rows.filter((row) => row.is_correct).length;
  const bonusPerTrial = state.payment?.correct_bonus_per_trial_usd || 0.10;
  const earnedBonus = (correct * bonusPerTrial).toFixed(2);

  $("progressBar").max = Math.max(1, total);
  $("progressBar").value = complete;
  $("studyProgressText").textContent = `${complete} / ${total} done`;
  $("progressBreakdown").textContent =
    `${complete} completed · ${remaining} remaining · ${correct} correct · bonus $${earnedBonus}` +
    (practiceTotal ? ` · practice ${Math.min(practiceDone, practiceTotal)}/${practiceTotal}` : "") +
    (attentionTotal ? ` · checks ${Math.min(attentionDone, attentionTotal)}/${attentionTotal}` : "");
  $("trialListSummary").textContent = `${total} paid trials · ${practiceTotal} practice · ${attentionTotal} checks`;
  renderPuzzleList($("trialPuzzleList"), state.order);
  updateElapsedClock();
}

function startTimer() {
  if (state.timerId) clearInterval(state.timerId);
  state.timerId = setInterval(updateElapsedClock, 1000);
}

function totalStudyElapsedMs() {
  if (!state.studyStartedAt) return 0;
  const end = state.studyEndedAt || performance.now();
  return Math.max(0, end - state.studyStartedAt);
}

function updateTotalElapsedClock() {
  $("totalElapsedText").textContent = `Total ${formatDuration(totalStudyElapsedMs())} / ${currentStudyValidWindowMinutes()}:00`;
}

function updateElapsedClock() {
  updateTotalElapsedClock();
  const puzzle = currentPuzzle();
  if (!state.studyStartedAt || !puzzle || (state.responseById.has(puzzle.id) && !state.awaitingNext)) {
    $("elapsedText").textContent = "00:00";
    return;
  }
  if (state.awaitingNext) {
    $("elapsedText").textContent = state.awaitingNextLabel || "Recorded";
    return;
  }
  const elapsed = performance.now() - state.trialStartedAt;
  if (isQualityPuzzle(puzzle)) {
    $("elapsedText").textContent = `${isPracticePuzzle(puzzle) ? "Practice" : "Check"} ${formatDuration(elapsed)}`;
    $("giveUpBtn").disabled = true;
    $("giveUpBtn").classList.add("hidden");
    return;
  }
  if (elapsed >= TRIAL_TIMEOUT_MS) {
    recordResponse({ timeout: true });
    return;
  }
  $("elapsedText").textContent = `Trial ${formatDuration(elapsed)} / 02:00`;
  const skipReady = elapsed >= SKIP_DELAY_MS;
  $("giveUpBtn").disabled = !skipReady;
  if (skipReady) {
    $("giveUpBtn").textContent = "I don't know";
  } else {
    const remaining = Math.max(0, Math.ceil((SKIP_DELAY_MS - elapsed) / 1000));
    $("giveUpBtn").textContent = `I don't know (${remaining}s)`;
  }
}

function formatDuration(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function markFirstAction() {
  if (state.studyStartedAt && state.trialStartedAt && !state.firstActionAt) {
    state.firstActionAt = performance.now();
  }
}

function orderedResponses() {
  return state.order.map((puzzle) => state.responseById.get(puzzle.id)).filter(Boolean);
}

function finalizedResponses() {
  const rows = orderedResponses().filter((row) => !row.is_practice && !row.is_attention_check);
  const badCount = rows.filter((row) => row.bad_record).length;
  const excluded = badCount >= BAD_RECORD_EXCLUSION_THRESHOLD;
  return rows.map((row) => ({
    ...row,
    participant_break_count: state.breakCount,
    participant_total_break_time_ms: Math.round(state.totalBreakTimeMs),
    participant_bad_record_count: badCount,
    participant_excluded_by_bad_record_rule: excluded,
    quota_counted_after_exclusion: !excluded && row.quota_counted,
  }));
}

function latestRecordedTrialEndIso() {
  const rows = orderedResponses();
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    if (rows[index]?.trial_end_time_iso) return rows[index].trial_end_time_iso;
  }
  return new Date().toISOString();
}

function responsesJsonl() {
  const rows = finalizedResponses();
  if (!rows.length) return "";
  return rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
}

async function finish() {
  if (state.breakStartedAt) resumeBreak();
  try {
    await finishBackendSession();
  } catch (error) {
    const errorMessage = String(error.message || "");
    const limitMatch = errorMessage.match(/^study_time_limit_exceeded:(\d+)/);
    const validWindowMinutes = limitMatch ? limitMatch[1] : currentStudyValidWindowMinutes();
    const message = errorMessage.startsWith("study_time_limit_exceeded")
      ? `This session exceeded the ${validWindowMinutes}-minute validity window and cannot receive a completion code.`
      : `Could not save completion: ${error.message || "backend error"}. Please retry Finish.`;
    setStatus(message, "warn");
    $("nextBtn").disabled = false;
    return;
  }
  if (state.timerId) clearInterval(state.timerId);
  state.studyEndedAt = performance.now();
  state.timerId = null;
  updateTotalElapsedClock();
  renderStudyProgress();
  $("trial").classList.add("hidden");
  $("breakPanel").classList.add("hidden");
  $("done").classList.remove("hidden");
  document.body.classList.remove("study-active");
  setStudyNavigationLocked(false);
  const correct = finalizedResponses().filter((row) => row.is_correct).length;
  const bonus = (correct * (state.payment?.correct_bonus_per_trial_usd || 0.10)).toFixed(2);
  $("doneSummary").textContent = `Assigned set complete. Correct trials: ${correct}. Current bonus: $${bonus}.`;
  if (state.completionCode) {
    $("completionCodeBox").classList.remove("hidden");
    $("completionCodeBox").textContent = `Completion code: ${state.completionCode}`;
  } else {
    $("completionCodeBox").classList.add("hidden");
  }
}


function takeBreak() {
  if (!state.awaitingNext || state.breakStartedAt) return;
  state.breakStartedAt = performance.now();
  $("trial").classList.add("hidden");
  $("breakPanel").classList.remove("hidden");
}

function resumeBreak() {
  if (state.breakStartedAt) {
    state.totalBreakTimeMs += performance.now() - state.breakStartedAt;
    state.breakStartedAt = 0;
    state.breakCount += 1;
    state.pendingJustAfterBreak = true;
  }
  $("breakPanel").classList.add("hidden");
  if (!$("done").classList.contains("hidden")) return;
  $("trial").classList.remove("hidden");
}

function setStatus(message, tone = "") {
  $("statusText").textContent = message;
  $("statusText").className = tone ? `status ${tone}` : "status";
}

function findPoint(grid, char) {
  for (let r = 0; r < grid.length; r += 1) {
    const c = grid[r].indexOf(char);
    if (c !== -1) return [r, c];
  }
  throw new Error(`Grid has no ${char}`);
}

function last(items) {
  return items[items.length - 1];
}

function coordKey(coord) {
  return `${coord[0]},${coord[1]}`;
}

function sameCoord(a, b) {
  return Boolean(a && b && a[0] === b[0] && a[1] === b[1]);
}

function inBounds(grid, r, c) {
  return r >= 0 && c >= 0 && r < grid.length && c < grid[0].length;
}

function isAdjacent(a, b) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1;
}

function movesFromCoords(coords) {
  const moves = [];
  for (let i = 1; i < coords.length; i += 1) {
    const dr = coords[i][0] - coords[i - 1][0];
    const dc = coords[i][1] - coords[i - 1][1];
    if (dr === -1 && dc === 0) moves.push("U");
    else if (dr === 1 && dc === 0) moves.push("D");
    else if (dr === 0 && dc === -1) moves.push("L");
    else if (dr === 0 && dc === 1) moves.push("R");
  }
  return moves.join("");
}

function toOneBased(coord) {
  return [coord[0] + 1, coord[1] + 1];
}

function sortedOneBased(coords) {
  return coords
    .map(toOneBased)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

function coordinatesAnswer(coords) {
  return sortedOneBased(coords)
    .map(([r, c]) => `(${r},${c})`)
    .join(", ");
}

function validateCurrentAnswer(puzzle) {
  if (!puzzle || !state.currentUi) return { valid: false, complete: false };
  if (!hasCompleteAnswer(puzzle)) return { valid: false, complete: false };

  if (puzzle.puzzle_type === "arithmetic24") {
    return validateArithmeticAnswer(puzzle);
  }
  if (puzzle.puzzle_type === "maze") {
    const moves = currentRawAnswer();
    const solution = puzzle.solutions.find((candidate) => candidate.moves === moves);
    return solution
      ? { valid: true, complete: true, solution }
      : { valid: false, complete: true, message: "This reaches G, but it is not one of the shortest valid paths." };
  }
  if (puzzle.puzzle_type === "grid_placement") {
    const key = coordinateSetKey(sortedOneBased(state.currentUi.selected));
    const solution = puzzle.solutions.find((candidate) => coordinateSetKey(candidate.coordinates) === key);
    return solution
      ? { valid: true, complete: true, solution }
      : { valid: false, complete: true, message: "This placement violates the puzzle constraints." };
  }
  if (puzzle.puzzle_type === "minesweeper_lite" || puzzle.puzzle_type === "mini_sudoku") {
    const selected = toOneBased(state.currentUi.selectedCell);
    const solution = puzzle.solutions.find((candidate) => sameOneBased(candidate.coordinate, selected));
    return solution
      ? { valid: true, complete: true, solution }
      : { valid: false, complete: true, message: "That cell is not a valid answer for this puzzle." };
  }
  return { valid: true, complete: true };
}

function validateArithmeticAnswer(puzzle) {
  const numberTokens = state.currentUi.tokens.filter((token) => token.type === "number");
  if (numberTokens.length !== puzzle.machine_readable_instance.numbers.length) {
    return { valid: false, complete: false };
  }
  const exact = evaluateFractionExpression(state.currentUi.tokens);
  if (!exact.ok) {
    return { valid: false, complete: true, message: "The expression is incomplete or malformed." };
  }
  if (exact.value.n === 24n && exact.value.d === 1n) {
    const canonical = keyToString(exact.key);
    const solution = puzzle.solutions.find((candidate) => candidate.canonical_expression === canonical);
    if (solution) return { valid: true, complete: true, solution };
    return {
      valid: false,
      complete: true,
      message: "This makes 24, but it does not match a recorded solution class. Please use another expression.",
    };
  }
  return { valid: false, complete: true, message: `This expression equals ${formatFraction(exact.value)}, not 24.` };
}

function evaluateFractionExpression(tokens) {
  let index = 0;

  function peek() {
    return tokens[index];
  }

  function take() {
    const token = tokens[index];
    index += 1;
    return token;
  }

  function parseExpression() {
    let left = parseTerm();
    if (!left.ok) return left;
    while (peek()?.value === "+" || peek()?.value === "-") {
      const op = take().value;
      const right = parseTerm();
      if (!right.ok) return right;
      left = {
        ok: true,
        value: op === "+" ? addFrac(left.value, right.value) : subFrac(left.value, right.value),
        key: canonicalKey(op, left.key, right.key),
      };
    }
    return left;
  }

  function parseTerm() {
    let left = parseFactor();
    if (!left.ok) return left;
    while (peek()?.value === "*" || peek()?.value === "/") {
      const op = take().value;
      const right = parseFactor();
      if (!right.ok) return right;
      if (op === "/" && right.value.n === 0n) return { ok: false };
      left = {
        ok: true,
        value: op === "*" ? mulFrac(left.value, right.value) : divFrac(left.value, right.value),
        key: canonicalKey(op, left.key, right.key),
      };
    }
    return left;
  }

  function parseFactor() {
    const token = take();
    if (!token) return { ok: false };
    if (token.type === "number") return { ok: true, value: frac(BigInt(token.value), 1n), key: ["n", Number(token.value)] };
    if (token.value === "(") {
      const inner = parseExpression();
      if (!inner.ok || take()?.value !== ")") return { ok: false };
      return inner;
    }
    return { ok: false };
  }

  const parsed = parseExpression();
  if (!parsed.ok || index !== tokens.length) return { ok: false };
  return parsed;
}

function frac(n, d) {
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const g = gcd(absBigInt(n), absBigInt(d));
  return { n: n / g, d: d / g };
}

function addFrac(a, b) {
  return frac(a.n * b.d + b.n * a.d, a.d * b.d);
}

function subFrac(a, b) {
  return frac(a.n * b.d - b.n * a.d, a.d * b.d);
}

function mulFrac(a, b) {
  return frac(a.n * b.n, a.d * b.d);
}

function divFrac(a, b) {
  return frac(a.n * b.d, a.d * b.n);
}

function gcd(a, b) {
  while (b !== 0n) {
    [a, b] = [b, a % b];
  }
  return a || 1n;
}

function absBigInt(value) {
  return value < 0n ? -value : value;
}

function formatFraction(value) {
  return value.d === 1n ? String(value.n) : `${value.n}/${value.d}`;
}

function canonicalKey(op, left, right) {
  if (op === "+") {
    const [lp, ln] = sumParts(left);
    const [rp, rn] = sumParts(right);
    return makeSumKey([...lp, ...rp], [...ln, ...rn]);
  }
  if (op === "-") {
    const [lp, ln] = sumParts(left);
    const [rp, rn] = sumParts(right);
    return makeSumKey([...lp, ...rn], [...ln, ...rp]);
  }
  if (op === "*") {
    const [ln, ld] = productParts(left);
    const [rn, rd] = productParts(right);
    return makeProductKey([...ln, ...rn], [...ld, ...rd]);
  }
  if (op === "/") {
    const [ln, ld] = productParts(left);
    const [rn, rd] = productParts(right);
    return makeProductKey([...ln, ...rd], [...ld, ...rn]);
  }
  throw new Error(`Unknown operator ${op}`);
}

function sumParts(key) {
  return key?.[0] === "sum" ? [[...key[1]], [...key[2]]] : [[key], []];
}

function productParts(key) {
  return key?.[0] === "prod" ? [[...key[1]], [...key[2]]] : [[key], []];
}

function makeSumKey(positives, negatives) {
  const pos = positives.sort(keyCompare);
  const neg = negatives.sort(keyCompare);
  if (pos.length === 1 && neg.length === 0) return pos[0];
  return ["sum", pos, neg];
}

function makeProductKey(numerators, denominators) {
  const num = numerators.sort(keyCompare);
  const den = denominators.sort(keyCompare);
  if (num.length === 1 && den.length === 0) return num[0];
  return ["prod", num, den];
}

function keyCompare(a, b) {
  return keyToString(a).localeCompare(keyToString(b));
}

function keyToString(key) {
  const tag = key[0];
  if (tag === "n") return String(key[1]);
  if (tag === "sum") {
    const positives = key[1];
    const negatives = key[2];
    let expr = positives.length ? positives.map(keyToString).join("+") : "0";
    for (const child of negatives) expr += `-${keyToString(child)}`;
    return `(${expr})`;
  }
  if (tag === "prod") {
    const numerators = key[1];
    const denominators = key[2];
    const num = numerators.length ? numerators.map(keyToString).join("*") : "1";
    if (!denominators.length) return `(${num})`;
    let den = denominators.map(keyToString).join("*");
    if (denominators.length > 1) den = `(${den})`;
    return `(${num}/${den})`;
  }
  return "";
}

function coordinateSetKey(coords) {
  return JSON.stringify([...coords].map(([r, c]) => [Number(r), Number(c)]).sort((a, b) => a[0] - b[0] || a[1] - b[1]));
}

function sameOneBased(a, b) {
  return Boolean(a && b && Number(a[0]) === Number(b[0]) && Number(a[1]) === Number(b[1]));
}


const RETRYABLE_POST_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504]);
const SAVE_RETRY_DELAYS_MS = [1000, 2000, 4000];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryablePostFailure(response, data) {
  if (!response) return true;
  return RETRYABLE_POST_STATUSES.has(response.status) || data?.error === "internal_server_error";
}

async function postJson(path, payload, options = {}) {
  const retryDelaysMs = options.retryDelaysMs || [];
  let lastError = null;
  let lastResult = null;
  for (let attempt = 0; attempt <= retryDelaysMs.length; attempt += 1) {
    try {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      let data = null;
      try {
        data = await response.json();
      } catch {
        data = {};
      }
      lastResult = { response, data };
      if (!isRetryablePostFailure(response, data) || attempt === retryDelaysMs.length) {
        return lastResult;
      }
    } catch (error) {
      lastError = error;
      if (attempt === retryDelaysMs.length) {
        throw error;
      }
    }
    const delayMs = retryDelaysMs[attempt];
    if (typeof options.onRetry === "function") {
      options.onRetry({ attempt: attempt + 1, delayMs, path });
    }
    await sleep(delayMs);
  }
  if (lastResult) return lastResult;
  throw lastError || new Error("request failed");
}

function retrySaveOptions(label) {
  return {
    retryDelaysMs: SAVE_RETRY_DELAYS_MS,
    onRetry: ({ attempt, delayMs }) => {
      setStatus(`${label} temporarily failed. Retrying in ${Math.round(delayMs / 1000)}s (${attempt}/3)...`, "warn");
    },
  };
}

function registrationErrorMessage(errorCode) {
  const messages = {
    invalid_username: "Username can only use letters, numbers, and underscores.",
    username_exists: "This username is already registered.",
    prolific_pid_exists: "This Prolific participant is already registered for this study.",
    prolific_required: "Open this study from Prolific using the official study link.",
    invalid_access_token: "This study link is invalid or expired. Return to Prolific and reopen the study.",
    prolific_token_required: "This study link is missing Prolific's secure token. Return to Prolific and reopen the study.",
    prolific_token_invalid: "We could not verify this Prolific link. Return to Prolific and reopen the study.",
    prolific_token_expired: "This Prolific link expired. Return to Prolific and reopen the study.",
    prolific_token_claim_mismatch: "This Prolific link does not match your session. Return to Prolific and reopen the study.",
    prolific_token_signature_invalid: "We could not verify this Prolific link. Return to Prolific and reopen the study.",
    prolific_token_key_not_found: "We could not verify this Prolific link. Return to Prolific and reopen the study.",
    prolific_jwks_fetch_failed: "Study verification is temporarily unavailable. Please try again soon.",
    prolific_api_token_missing: "Study verification is not configured yet. Please contact the researcher.",
    prolific_submission_not_found: "We could not find your Prolific submission. Return to Prolific and reopen the study.",
    prolific_submission_mismatch: "This Prolific session does not match the study link. Return to Prolific and reopen the study.",
    prolific_submission_status_invalid: "Your Prolific submission is not currently eligible for registration.",
    prolific_verification_error: "We could not verify your Prolific session. Please try again soon.",
    invalid_tester_login: "Researcher test login failed.",
    invalid_tester_cohort: "Researcher test links can only use study=main or study=module.",
  };
  return messages[errorCode] || errorCode || "Registration failed.";
}

async function registerWithBackend(username) {
  try {
    const cohort = state.cohort;
    if (state.testerMode) {
      const { response, data } = await postJson("/api/tester/login", {
        tester_username: $("testerName").value.trim(),
        password: $("testerPassword").value,
        cohort,
        consent_at: state.consentAt,
      }, retrySaveOptions("Registering"));
      if (!response.ok || !data.ok) {
        const code = data.error || "invalid_tester_login";
        const error = new Error(registrationErrorMessage(code));
        error.registrationCode = code;
        throw error;
      }
      return data;
    }
    const { response, data } = await postJson("/api/register", {
      username,
      cohort,
      prolific_pid: state.prolificPid,
      prolific_study_id: state.prolificStudyId,
      prolific_session_id: state.prolificSessionId,
      access_token: ACCESS_TOKEN,
      prolific_token: PROLIFIC_TOKEN,
      consent_at: state.consentAt,
    }, retrySaveOptions("Registering"));
    if (!response.ok || !data.ok) {
      const code = data.error || "registration_failed";
      const error = new Error(registrationErrorMessage(code));
      error.registrationCode = code;
      throw error;
    }
    return data;
  } catch (error) {
    throw error;
  }
}

function applyBackendRegistration(backendRegistration, displayUsername) {
  const assignment = backendRegistration.assignment;
  state.participantId = backendRegistration.username || displayUsername;
  state.testerLogin = backendRegistration.tester_login || "";
  $("participantId").value = state.participantId;
  state.cohort = backendRegistration.cohort;
  state.studyValidWindowMinutes =
    Number(assignment.study?.study_valid_window_minutes) || studyValidWindowMinutesForCohort(state.cohort);
  state.backendMode = true;
  state.assignmentSource = "backend";
  state.registered = true;
  state.datasetPath = assignment.dataset_path;
  state.payment = assignment.payment || null;
  state.completionCode = assignment.completion_code || "";
  state.participantSessionToken = assignment.participant_session_token || "";
  state.assignmentVersion = assignment.assignment_version || "";
  state.assignmentHash = assignment.assignment_hash || "";
  state.assignmentMetaByPuzzle = buildAssignmentMeta(assignment);
  state.resuming = Boolean(assignment.resume);
  state.initialResponses = new Map(
    [...(assignment.existing_responses || []), ...(assignment.existing_quality_checks || [])].map((row) => [row.puzzle_id, row]),
  );
  if (!Array.isArray(assignment.public_puzzles) || !assignment.public_puzzles.length) {
    $("loadStatus").textContent = "Registration did not include the public puzzle payload. Please refresh and try again.";
    state.registered = false;
    state.backendMode = false;
    return false;
  }
  state.puzzles = assignment.public_puzzles;
  const byId = new Map(state.puzzles.map((puzzle) => [puzzle.id, puzzle]));
  state.plannedOrder = assignment.puzzle_ids.map((id) => byId.get(id)).filter(Boolean);
  $("loadStatus").textContent = state.resuming
    ? `${state.participantId} resumed. ${state.initialResponses.size} saved responses found; ${state.plannedOrder.length} puzzles assigned.`
    : `${state.participantId} registered. ${state.plannedOrder.length} puzzles assigned.`;
  updatePlannedOrder();
  return true;
}

function loadAdminTestAssignment() {
  let payload = null;
  try {
    payload = JSON.parse(sessionStorage.getItem(ADMIN_TEST_ASSIGNMENT_KEY) || "null");
  } catch {
    payload = null;
  }
  if (!payload || !payload.data || !payload.data.assignment) {
    $("loadStatus").textContent = "Admin login is missing. Open Admin from the study page and sign in again.";
    $("registerBtn").disabled = true;
    return;
  }
  const expectedStudy = state.cohort === "module" ? "module" : "main";
  if (payload.data.cohort !== expectedStudy) {
    $("loadStatus").textContent = "Admin login does not match this study type. Return to the Admin link.";
    $("registerBtn").disabled = true;
    return;
  }
  $("consentCheck").checked = true;
  state.consentAt = payload.consent_at || new Date().toISOString();
  if (applyBackendRegistration(payload.data, payload.data.username || "admin_test")) {
    $("participantMeta").textContent = `Prolific-style test session: ${state.participantId}`;
    if (state.autoStartAdminTest) {
      startAssignedSet();
    }
  }
}

async function registerParticipant() {
  const username = state.testerMode ? "researcher_test_pending" : $("participantId").value.trim();
  if (state.formalMode && !$("consentCheck").checked) {
    $("loadStatus").textContent = "Please confirm the study information before registering.";
    return;
  }
  if (state.formalMode && !state.consentAt) {
    state.consentAt = new Date().toISOString();
  }
  if (state.testerMode && (!$("testerName").value.trim() || !$("testerPassword").value)) {
    $("loadStatus").textContent = "Enter researcher username and password.";
    $("testerName").focus();
    return;
  }
  if (!state.testerMode && !USERNAME_RE.test(username)) {
    $("loadStatus").textContent = "Username can only use letters, numbers, and underscores.";
    $("participantId").focus();
    return;
  }
  $("loadStatus").textContent = "Registering...";

  let backendRegistration = null;
  try {
    backendRegistration = await registerWithBackend(username);
  } catch (error) {
    $("loadStatus").textContent = error.message || "Registration failed.";
    return;
  }
  if (backendRegistration) {
    applyBackendRegistration(backendRegistration, username);
    return;
  }

  $("loadStatus").textContent = "Could not reach the study server. Please refresh and try again.";
}

async function sendResponseToBackend(row) {
  if (!state.backendMode) return { ok: true };
  const { response, data } = await postJson("/api/response", row, retrySaveOptions("Saving answer"));
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "save failed");
  }
  return data;
}

async function sendQualityCheckToBackend(row) {
  if (!state.backendMode) return { ok: true };
  const { response, data } = await postJson("/api/quality_check", row, retrySaveOptions("Saving check"));
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "quality-check save failed");
  }
  return data;
}

async function finishBackendSession() {
  if (!state.backendMode) return null;
  const { response, data } = await postJson("/api/finish", {
    username: state.participantId,
    participant_session_token: state.participantSessionToken || "",
    study_start_time_iso: state.studyStartedWallMs ? new Date(state.studyStartedWallMs).toISOString() : "",
    study_end_time_iso: latestRecordedTrialEndIso(),
    study_elapsed_ms: state.studyStartedAt ? Math.round(performance.now() - state.studyStartedAt) : null,
    study_valid_window_minutes: currentStudyValidWindowMinutes(),
  }, retrySaveOptions("Saving completion"));
  if (response.ok && data.ok) {
    state.completionCode = data.completion_code || state.completionCode;
    if (data.payment) state.payment = data.payment;
    return data;
  }
  throw new Error(data.error || "finish failed");
}

async function bootstrap() {
  configureFormalMode();
  $("startBtn").disabled = true;
  try {
    await loadQualityChecks();
  } catch (error) {
    $("loadStatus").textContent = error.message;
    $("registerBtn").disabled = true;
    return;
  }
  if (state.adminTestSession) {
    loadAdminTestAssignment();
    return;
  }
  $("loadStatus").textContent = "Ready to register.";
  updatePlannedOrder();
}


function startAssignedSet() {
  if (!state.registered) {
    $("loadStatus").textContent = "Register your username first.";
    return false;
  }
  state.realOrder = state.plannedOrder;
  state.practiceOrder = state.formalMode
    ? PRACTICE_PUZZLES.slice(0, 3).filter((puzzle) => !state.initialResponses.has(puzzle.id))
    : [];
  const attentionCount = state.formalMode ? attentionCheckCountForCohort() : 0;
  state.attentionOrder = ATTENTION_PUZZLES.slice(0, attentionCount);
  state.order = [...state.practiceOrder, ...insertAttentionChecks(state.realOrder, state.attentionOrder)];
  if (!state.realOrder.length) {
    $("loadStatus").textContent = "Select at least one puzzle.";
    return false;
  }
  state.responseById = new Map(state.initialResponses);
  const firstUnanswered = state.order.findIndex((puzzle) => !state.responseById.has(puzzle.id));
  const allComplete = firstUnanswered === -1;
  state.index = firstUnanswered >= 0 ? firstUnanswered : state.order.length - 1;
  state.studyStartedAt = performance.now();
  state.studyStartedWallMs = Date.now();
  state.studyEndedAt = 0;
  state.breakStartedAt = 0;
  state.breakCount = 0;
  state.totalBreakTimeMs = 0;
  state.pendingJustAfterBreak = false;
  state.currentTrialAfterBreak = false;
  $("setup").classList.add("hidden");
  $("trial").classList.remove("hidden");
  $("done").classList.add("hidden");
  document.body.classList.add("study-active");
  setStudyNavigationLocked(true);
  updateTotalElapsedClock();
  startTimer();
  if (allComplete) {
    finish();
    return true;
  }
  startTrial();
  return true;
}

$("startBtn").addEventListener("click", startAssignedSet);

$("registerBtn").addEventListener("click", registerParticipant);
$("consentCheck").addEventListener("change", () => {
  if (state.formalMode) {
    $("registerBtn").disabled = !$("consentCheck").checked;
  }
});
$("submitBtn").addEventListener("click", () => {
  recordInteraction("submit_click", { target: "submit_final_answer" });
  submitCurrentAnswer();
});
$("giveUpBtn").addEventListener("click", () => {
  recordInteraction("give_up_click", { target: "give_up" });
  recordResponse({ gaveUp: true });
});
$("nextBtn").addEventListener("click", goToNextAfterCorrect);
$("breakBtn").addEventListener("click", takeBreak);
$("resumeBtn").addEventListener("click", resumeBreak);
$("refreshListBtn").addEventListener("click", updatePlannedOrder);
["visualArea", "answerArea"].forEach((id) => {
  $(id).addEventListener("click", markFirstAction, true);
  $(id).addEventListener("click", (event) => recordAreaClick(event, id));
  $(id).addEventListener("keydown", markFirstAction, true);
});

bootstrap();

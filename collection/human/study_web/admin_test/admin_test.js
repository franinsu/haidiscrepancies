const PARAMS = new URLSearchParams(window.location.search);
const STUDY = PARAMS.get("study") === "module" ? "module" : "main";
const STORAGE_KEY = "reasoningStudyAdminTestAssignmentV1";

sessionStorage.removeItem(STORAGE_KEY);

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }
  return { response, data };
}

function freshTesterSessionId() {
  const bytes = new Uint32Array(2);
  crypto.getRandomValues(bytes);
  return `${Date.now().toString(36)}_${Array.from(bytes, (value) => value.toString(36)).join("_")}`;
}

document.getElementById("adminTestForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = document.getElementById("testerName").value.trim();
  const password = document.getElementById("testerPassword").value;
  const button = document.getElementById("submitBtn");
  if (!name || !password) return;
  button.disabled = true;
  setStatus("");
  try {
    const { response, data } = await postJson("/api/tester/login", {
      tester_username: name,
      password,
      cohort: STUDY,
      tester_session_id: freshTesterSessionId(),
      consent_at: new Date().toISOString(),
    });
    if (!response.ok || !data.ok || !data.assignment) {
      setStatus("Invalid name or password.");
      button.disabled = false;
      return;
    }
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        study: STUDY,
        consent_at: new Date().toISOString(),
        data,
      }),
    );
    window.location.href = `../app/?formal=1&study=${STUDY}&admin_test_session=1&autostart=1`;
  } catch {
    setStatus("Could not start. Try again.");
    button.disabled = false;
  }
});

const $ = (id) => document.getElementById(id);
const nextPath = new URLSearchParams(window.location.search).get("next") || "";

async function request(path, payload) {
  const options = payload === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

async function login() {
  try {
    await request("/api/admin/login", { username: $("adminName").value.trim(), password: $("adminPassword").value });
    $("adminPassword").value = "";
    if (nextPath.startsWith("/") && !nextPath.startsWith("//") && !nextPath.startsWith("/study_web/admin")) {
      window.location.href = nextPath;
      return;
    }
    const status = await request("/api/admin/status");
    $("collectionStatus").textContent = `${status.counts.participants} participants · ${status.counts.responses} responses · ${status.counts.quality_checks} quality checks`;
    $("loginPanel").classList.add("hidden");
    $("adminPanel").classList.remove("hidden");
    $("loginStatus").textContent = "";
  } catch (error) { $("loginStatus").textContent = error.message; }
}

async function exportRecords() {
  $("exportBtn").disabled = true;
  $("actionStatus").textContent = "Preparing export…";
  try {
    const table = $("table").value;
    const cohort = $("cohort").value;
    let offset = 0;
    const lines = [];
    do {
      const query = new URLSearchParams({ table, cohort, offset, limit: 1000 });
      const page = await request(`/api/admin/export_rows?${query}`);
      lines.push(...page.rows.map((row) => JSON.stringify(row)));
      offset = page.next_offset;
    } while (offset !== null);
    const blob = new Blob([lines.join("\n") + (lines.length ? "\n" : "")], { type: "application/jsonl" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${table}.jsonl`;
    link.click();
    URL.revokeObjectURL(url);
    $("actionStatus").textContent = `Downloaded ${lines.length} ${cohort} records.`;
  } catch (error) { $("actionStatus").textContent = error.message; }
  finally { $("exportBtn").disabled = false; }
}

$("loginBtn").addEventListener("click", login);
$("adminPassword").addEventListener("keydown", (event) => { if (event.key === "Enter") login(); });
$("exportBtn").addEventListener("click", exportRecords);
$("backupBtn").addEventListener("click", async () => {
  try {
    const result = await request("/api/admin/backup", {});
    $("actionStatus").textContent = `Database backup saved: ${result.backup_path}`;
  } catch (error) { $("actionStatus").textContent = error.message; }
});
$("logoutBtn").addEventListener("click", async () => {
  try {
    await request("/api/admin/logout", {});
    $("adminPanel").classList.add("hidden");
    $("loginPanel").classList.remove("hidden");
  } catch (error) { $("actionStatus").textContent = error.message; }
});

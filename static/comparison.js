let comparison_id = null;

function startComparison() {
  fetch("/comparison/create", { method: "POST" })
    .then(r => r.json())
    .then(d => comparison_id = d.comparison_id);
}

function uploadCandidates() {
  const input = document.getElementById("candidateInput");
  if (!input.files.length) return alert("Select files");

  const fd = new FormData();
  for (let f of input.files) fd.append("files[]", f);

  fetch(`/comparison/${comparison_id}/upload`, {
    method: "POST",
    body: fd
  });
}

function analyzeAndGraph() {
  fetch(`/comparison/${comparison_id}/analyze`)
    .then(r => r.json())
    .then(response => {
      const c = response.data.candidates;

      const names = c.map(x => x.name);
      const scores = c.map(x => x.relevance_score);
      const exp = c.map(x => x.years_experience);

      if (window.myChart) window.myChart.destroy();

      window.myChart = new Chart(document.getElementById("comparisonChart"), {
        type: "bar",
        data: {
          labels: names,
          datasets: [
            { label: "Relevance Score", data: scores },
            { label: "Years Experience", data: exp }
          ]
        },
        options: { scales: { y: { beginAtZero: true, max: 100 } } }
      });

      document.getElementById("analysisSummary").innerText = response.data.summary;
    });
}
async function runComparison() {
    try {
        const input = document.getElementById("compareInput");
        const files = input.files;

        if (!files || files.length < 2) {
            alert("Please select at least 2 resumes.");
            return;
        }

        const sessionRes = await fetch("/comparison/create", { method: "POST" });
        const sessionData = await sessionRes.json();
        const comparison_id = sessionData.comparison_id;

        const formData = new FormData();
        for (let f of files) formData.append("files[]", f);

        await fetch(`/comparison/${comparison_id}/upload`, {
            method: "POST",
            body: formData
        });

        const res = await fetch(`/comparison/${comparison_id}/analyze`);

        if (!res.ok) {
            const text = await res.text();
            console.error("Analyze failed:", text);
            alert("Analyze failed — see console");
            return;
        }

        const data = await res.json();
        console.log("ANALYZE DATA:", data);

        const chartContainer = document.getElementById("chartContainer");
        chartContainer.style.display = "flex";


        const canvas = document.getElementById("comparisonChart");
        canvas.style.height = "300px";
        canvas.style.width = "100%";

        const ctx = canvas.getContext("2d");

        if (window.comparisonChart && typeof window.comparisonChart.destroy === "function") {
            window.comparisonChart.destroy();
        }

        const candidates = data.data.candidates;

        window.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: candidates.map(c => c.name),
                datasets: [
                    { label: 'Relevance Score', data: candidates.map(c => c.relevance_score) },
                    { label: 'Years Experience', data: candidates.map(c => c.years_experience), type: 'line' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100, ticks: { stepSize: 10 } }
               },
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        });

    } catch (err) {
        console.error("Comparison error:", err);
        alert("Failed to compare. See console.");
    }
}
window.runComparison = runComparison;

// // --- GLOBAL STATE ---
// var currentResumeId = null;       // For single-resume chat
// var activeComparisonIds = [];     // For multi-resume chat
// var isComparisonMode = false;     // Toggle for logic switching

// // --- 1. SINGLE RESUME UPLOAD (Your main js code updated) ---
// function upload() {
//     const fileInput = document.getElementById('fileInput'); // Use your actual ID
//     const status = document.getElementById("upload-status") || { innerText: "" };

//     if (!fileInput.files.length) {
//         Swal.fire('Error', 'Please select a file first!', 'error');
//         return;
//     }

//     const formData = new FormData();
//     formData.append("file", fileInput.files[0]);
//     status.innerText = "Processing...";

//     fetch("/upload", { method: "POST", body: formData })
//     .then(r => r.json())
//     .then(data => {
//         if (data.error) {
//             status.innerText = "Error: " + data.error;
//         } else {
//             // Enter Single Chat Mode
//             isComparisonMode = false;
//             currentResumeId = data.resume_id;
            
//             // UI Updates
//             status.innerText = "✅ Ready!";
//             document.getElementById("headerTitle").innerText = data.title;
//             document.getElementById("chat").innerHTML = ""; 
//             addMessage("Bot", `**${data.title}** is ready. What would you like to know?`);
            
//             // Enable Input
//             document.getElementById("questionInput").disabled = false;
//             document.getElementById("sendBtn").disabled = false;
//             loadHistory(); // Refresh sidebar
//         }
//     });
// }

// // --- 2. MULTI-RESUME "COMPARE" LOGIC ---
// async function runComparison() {
//     const fileInput = document.getElementById('compareInput');
//     const files = fileInput.files;

//     if (files.length < 2) {
//         Swal.fire('Info', 'Select at least 2 resumes to compare.', 'info');
//         return;
//     }

//     // Enter Comparison Mode
//     activeComparisonIds = [];
//     isComparisonMode = true;
    
//     // Upload all files and collect IDs
//     for (let file of files) {
//         const formData = new FormData();
//         formData.append("file", file);
//         const r = await fetch("/upload", { method: "POST", body: formData });
//         const data = await r.json();
//         activeComparisonIds.push(data.resume_id);
//     }

//     // UI Updates
//     toggleModal(false);
//     document.getElementById("headerTitle").innerText = "Comparing " + files.length + " Candidates";
//     document.getElementById("chat").innerHTML = "";
//     addMessage("Bot", "I have processed all resumes. You can now ask me to **compare skills**, **rank them**, or **generate a chart**!");
    
//     document.getElementById("questionInput").disabled = false;
//     document.getElementById("sendBtn").disabled = false;
// }

// // --- 3. THE SMART CHAT FUNCTION ---
// function askQuestion() {
//     const qInput = document.getElementById("questionInput");
//     const q = qInput.value.trim();
//     if (!q) return;

//     addMessage("You", q);
//     qInput.value = "";

//     // Payload chooses between single ID or the List of comparison IDs
//     const payload = {
//         question: q,
//         resume_id: isComparisonMode ? null : currentResumeId,
//         resume_ids: isComparisonMode ? activeComparisonIds : []
//     };

//     fetch("/ask", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(payload)
//     })
//     .then(r => r.json())
//     .then(data => {
//         // Here we handle the DYNAMIC CHART if the AI provides one
//         addMessage("Bot", data.answer, data.chart || null);
//     });
// }

// // --- 4. DYNAMIC MESSAGE & CHART RENDERER ---
// function addMessage(sender, text, chartConfig = null) {
//     const chatDiv = document.getElementById("chat");
//     const msgDiv = document.createElement("div");
//     msgDiv.className = "msg " + sender;
    
//     // Render text with Markdown
//     msgDiv.innerHTML = `<div>${marked.parse(text)}</div>`;

//     // If AI sent chart data, inject a canvas
//     if (chartConfig) {
//         const canvasId = "chart-" + Date.now();
//         msgDiv.innerHTML += `<div style="height:250px; margin-top:10px;"><canvas id="${canvasId}"></canvas></div>`;
//         chatDiv.appendChild(msgDiv); // Must append before creating Chart
        
//         new Chart(document.getElementById(canvasId), {
//             type: chartConfig.type || 'bar',
//             data: {
//                 labels: chartConfig.labels,
//                 datasets: chartConfig.datasets
//             },
//             options: { responsive: true, maintainAspectRatio: false }
//         });
//     } else {
//         chatDiv.appendChild(msgDiv);
//     }

//     chatDiv.scrollTop = chatDiv.scrollHeight;
// }
// function toggleModal(show) {
//     const modal = document.getElementById("compareModal");
//     modal.style.display = show ? "flex" : "none";
// }

// window.toggleModal = toggleModal;

// static/js/main.js

// static/js/main.js

function uploadResume() {
    let input = document.getElementById("resumeInput");
    // Handle case where ID might differ in your HTML
    if (!input) input = document.querySelector('input[type="file"]');
    
    if (!input || !input.files.length) return alert("Please select a file first");

    let formData = new FormData();
    formData.append("resume", input.files[0]);

    // Use a helper if available, or just console log
    if(typeof addMsg === "function") addMsg("System", "Uploading...", "bot");

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if(data.error) {
            alert("Error: " + data.error);
        } else {
            // FIX: This line fixes the "undefined uploaded" message
            let fileName = data.filename || "Resume";
            
            // Update the UI text if the element exists
            let statusDiv = document.querySelector(".upload-status") || document.getElementById("uploadStatus"); 
            if(statusDiv) statusDiv.innerText = fileName + " uploaded.";
            
            // Also notify in chat
            if(typeof addMsg === "function") addMsg("System", fileName + " uploaded successfully!", "bot");
        }
    })
    .catch(err => console.error("Upload failed", err));
}

function askQuestion() {
    let input = document.getElementById("question");
    if (!input) input = document.querySelector('input[type="text"]');
    
    let q = input.value;
    if (!q) return;

    if(typeof addMsg === "function") addMsg("You", q, "user");
    input.value = "";

    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
    })
    .then(r => r.json())
    .then(data => {
        if(typeof addMsg === "function") addMsg("Bot", data.answer, "bot");
    })
    .catch(err => console.error("Chat error", err));
}
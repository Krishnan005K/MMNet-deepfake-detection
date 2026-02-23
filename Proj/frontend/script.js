const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("video");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("result-box");
const historyDiv = document.getElementById("history-list");

let selectedFile = null;
let timerInterval;
let startTime;
// ---------------------------
// CLICK TO SELECT
// ---------------------------
dropZone.onclick = () => fileInput.click();

// ---------------------------
// FILE SELECTED
// ---------------------------
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        selectedFile = fileInput.files[0];
        updateDropZoneUI(selectedFile.name);
    }
});

// ---------------------------
// DRAG EVENTS
// ---------------------------
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");

    selectedFile = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;

    updateDropZoneUI(selectedFile.name);
});

// ---------------------------
// UPDATE DROP ZONE AFTER FILE ADDED
// ---------------------------
function updateDropZoneUI(fileName) {
    dropZone.innerHTML = `
        <p style="font-size:16px;">✅ File Ready for Analysis</p>
        <span style="color:#22c55e;">${fileName}</span>
        <p style="font-size:12px; margin-top:8px; color:#94a3b8;">
            Click again to change file
        </p>
        <input type="file" id="video" accept="video/*" hidden>
    `;

    dropZone.style.borderColor = "#22c55e";

    // Reattach file input
    const newInput = dropZone.querySelector("input");
    newInput.addEventListener("change", (e) => {
        selectedFile = e.target.files[0];
        updateDropZoneUI(selectedFile.name);
    });

    dropZone.onclick = () => newInput.click();
}
// ---------------------------
// TIMER (OPTIONAL)
// ---------------------------
function startTimer() {
    startTime = Date.now();

    timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        resultBox.innerHTML = `
            Processing...  Time Elapsed:
            <span style="font-size:14px; color:#94a3b8;">
                 ${elapsed}s
            </span>
        `;
    }, 100);
}

function stopTimer() {
    clearInterval(timerInterval);
}
// ---------------------------
// ANALYZE BUTTON
// ---------------------------
analyzeBtn.onclick = async () => {

    if (!selectedFile) {
        alert("Please select a video first");
        return;
    }

    const form = new FormData();
    form.append("video", selectedFile);

   //resultBox.innerHTML = "Processing... ⏳ Please wait";
    startTimer();

    try {
        const res = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: form
        });

        const data = await res.json();

       //resultBox.innerText = data.raw_report;
        
        stopTimer();
      resultBox.innerText = data.raw_report;
        // Refresh history after success
        loadHistory();

    } catch (err) {
        stopTimer();
        resultBox.innerHTML = "❌ Error processing video";
    }
};

// ---------------------------
// LOAD HISTORY
// ---------------------------
async function loadHistory() {
    try {
        const res = await fetch("http://127.0.0.1:8000/reports");
        const data = await res.json();

        historyDiv.innerHTML = "";

        if (data.reports.length === 0) {
            historyDiv.innerHTML = "<p class='no-history'>No processing history yet</p>";
            return;
        }

        data.reports.forEach(report => {
            const div = document.createElement("div");
            div.className = "history-item";
            div.innerText = report.filename;

            div.onclick = async () => {
                loadReport(report.filename);
            };

            historyDiv.appendChild(div);
        });

    } catch (err) {
        historyDiv.innerHTML = "<p class='no-history'>Error loading history</p>";
    }
}

// ---------------------------
// LOAD SPECIFIC REPORT
// ---------------------------
const modal = document.getElementById("reportModal");
const modalContent = document.getElementById("modalReportContent");
const modalTitle = document.getElementById("modalTitle");
const closeModalBtn = document.getElementById("closeModal");
const downloadBtn = document.getElementById("downloadBtn");

let currentReportFile = "";

// Open report in popup
async function loadReport(filename) {
    try {
        
        const res = await fetch("http://127.0.0.1:8000/reports/" + filename);
        const text = await res.text();

        modalTitle.innerText = filename;
        modalContent.innerText = text;
        currentReportFile = filename;

        modal.style.display = "flex";

    } catch {
        alert("Unable to load report");
    }
}

// Close modal
closeModalBtn.onclick = () => {
    modal.style.display = "none";
};

// Close when clicking outside
window.onclick = (e) => {
    if (e.target === modal) {
        modal.style.display = "none";
    }
};

// Download button
downloadBtn.onclick = () => {
    window.open("http://127.0.0.1:8000/reports/" + currentReportFile, "_blank");
};

// ---------------------------
// INITIAL LOAD
// ---------------------------
loadHistory();
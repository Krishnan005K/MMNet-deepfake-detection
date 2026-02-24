// ==========================================
// ELEMENTS
// ==========================================
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("video");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("result-box");
const historyDiv = document.getElementById("history-list");

let selectedFile = null;
let timerInterval;
let startTime;


// ==========================================
// FILE SELECTION (CLICK)
// ==========================================
dropZone.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        updateDropZoneUI(selectedFile.name);
    }
});


// ==========================================
// DRAG & DROP
// ==========================================
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

    if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        fileInput.files = e.dataTransfer.files;
        updateDropZoneUI(selectedFile.name);
    }
});


// ==========================================
// UPDATE DROPZONE UI
// ==========================================
function updateDropZoneUI(fileName) {
    dropZone.innerHTML = `
        <p style="font-size:16px;">✅ File Ready</p>
        <span style="color:#22c55e;">${fileName}</span>
        <p style="font-size:12px; margin-top:8px; color:#94a3b8;">
            Click to change file
        </p>
        <input type="file" id="video" accept="video/*" hidden>
    `;

    dropZone.style.borderColor = "#22c55e";

    const newInput = dropZone.querySelector("input");
    newInput.addEventListener("change", (e) => {
        selectedFile = e.target.files[0];
        updateDropZoneUI(selectedFile.name);
    });

    dropZone.addEventListener("click", () => newInput.click());
}


// ==========================================
// TIMER
// ==========================================
function startTimer() {
    startTime = Date.now();

    timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        resultBox.innerHTML = `
            Processing... ⏳ 
            <span style="color:#94a3b8; font-size:14px;">
                ${elapsed}s
            </span>
        `;
    }, 100);
}

function stopTimer() {
    clearInterval(timerInterval);
}


// ==========================================
// ANALYZE BUTTON
// ==========================================
analyzeBtn.addEventListener("click", async () => {

    if (!selectedFile) {
        alert("Please select a video first");
        return;
    }

    const formData = new FormData();
    formData.append("video", selectedFile);

    startTimer();

    try {
        const response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        stopTimer();

        resultBox.innerText = data.raw_report;

        loadHistory();

    } catch (error) {
        stopTimer();
        resultBox.innerHTML = "❌ Error processing video";
        console.error(error);
    }
});


// ==========================================
// LOAD REPORT HISTORY
// ==========================================
async function loadHistory() {
    try {
        const res = await fetch("http://127.0.0.1:8000/reports");
        const data = await res.json();

        historyDiv.innerHTML = "";

        if (!data.reports || data.reports.length === 0) {
            historyDiv.innerHTML =
                "<p class='no-history'>No processing history yet</p>";
            return;
        }

        // Group reports by folder
        const grouped = {};

        data.reports.forEach(report => {
            if (!grouped[report.folder]) {
                grouped[report.folder] = [];
            }
            grouped[report.folder].push(report.filename);
        });

        // Render folders
        Object.keys(grouped).forEach(folder => {

            const folderDiv = document.createElement("div");
            folderDiv.className = "history-folder";

            const folderTitle = document.createElement("h4");
            folderTitle.innerText = `📁 ${folder}`;
            folderDiv.appendChild(folderTitle);

            grouped[folder].forEach(file => {

                const fileDiv = document.createElement("div");
                fileDiv.className = "history-item";

                let icon = "📄";
                if (file.endsWith(".pdf")) icon = "📑";
                if (file.endsWith(".csv")) icon = "🗂️";
                if (file.endsWith(".txt")) icon = "📄";

                fileDiv.innerText = `${icon} ${file}`;

                fileDiv.addEventListener("click", () => {
                    window.open(
                        `http://127.0.0.1:8000/reports/${folder}/${file}`,
                        "_blank"
                    );
                });

                folderDiv.appendChild(fileDiv);
            });

            historyDiv.appendChild(folderDiv);
        });

    } catch (error) {
        historyDiv.innerHTML =
            "<p class='no-history'>Error loading history</p>";
        console.error(error);
    }
}


// ==========================================
// INITIAL LOAD
// ==========================================
loadHistory();
var resume_id = null; // Global storage for the session

function upload() {
    const fileInput = document.querySelector('input[type="file"]');
    const status = document.getElementById("upload-status");

    // 1. Validation
    if (!fileInput.files.length) {
        alert("Please select a file first!");
        return;
    }

    status.innerText = "Uploading & Processing...";

    // 2. Wrap the file in a FormData object (Fixes the "form is undefined" crash)
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // 3. Send to Flask
    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            status.innerText = "Error: " + data.error;
        } else {
            // Success: Store the ID so the AI knows which resume to read
            resume_id = data.resume_id;
            status.innerText = "✅ PDF Processed! You can now chat.";
            console.log("Resume ID set:", resume_id);
        }
    })
    .catch(err => {
        console.error(err);
        status.innerText = "System Error. Check console.";
    });
}
import { startCamera, getDroidCamId } from "./camera.js";

const video = document.getElementById("video");

async function init() {
  // 1. Try to find DroidCam automatically
  let selectedCameraId = await getDroidCamId();

  // 2. If not found, ask user for permission to find default camera
  if (!selectedCameraId) {
    console.log("DroidCam not found, using default.");
    // This allows the browser to ask for permission so labels become visible
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    // Stop the stream immediately, we just needed permissions
    stream.getTracks().forEach(t => t.stop());
    
    // Try finding it again now that we have permissions
    selectedCameraId = await getDroidCamId();
  }

  // 3. Start the camera with the specific ID
  if (selectedCameraId) {
    startCamera(selectedCameraId, video);
  } else {
    // Fallback if DroidCam is still not found
    console.log("Starting default camera");
    startCamera(undefined, video); 
  }
}

init();
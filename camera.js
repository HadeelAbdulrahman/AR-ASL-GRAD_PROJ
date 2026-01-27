export async function startCamera(deviceId, videoElement) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: { exact: deviceId } }
  });
  videoElement.srcObject = stream;
}

export async function getDroidCamId() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  
  // Look for a video input device with "DroidCam" in the label
  const droidCam = devices.find(device => 
    device.kind === 'videoinput' && device.label.includes("DroidCam")
  );

  return droidCam ? droidCam.deviceId : null;
}
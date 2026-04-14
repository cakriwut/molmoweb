/**
 * 
 * @param {(...args: any[]) => boolean} conditionFunction 
 * @param {number | undefined} maxRetries 
 * @returns {Promise<void>}
 */
function waitUntil(conditionFunction, maxRetries = 3) {
    let retries = 0;

    const poll = (resolve) => {
        if (conditionFunction() || retries >= maxRetries) {
            resolve();
        } else {
            retries++;
            setTimeout(() => poll(resolve), 100);
        }
    }
    
    return new Promise(poll);
}

/**
 * Converts a Blob object to a Base64 string
 * @param {Blob} blob 
 * @returns {Promise<string | ArrayBuffer | null>}
 */
const convertBlobToBase64 = (blob) => {
    return new Promise(resolve => {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
            const base64data = reader.result; // The Base64-encoded data
            resolve(base64data); // Resolves with the Base64 data
        };
    });
};

// Listen for messages from the Chrome extension background or content script
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.target === 'offscreen') {
        switch (message.type) {
            case 'start-recording':
                startRecording(message.data);
                break;
            case 'stop-recording':
                stopRecording().then(sessionRecording => {
                    sendResponse(sessionRecording)
                });
                return true;
            default:
                throw new Error('Unrecognized message:', message.type); // Handle unrecognized message types
        }
    }
});

// Initialize variables for the MediaRecorder and video data
/**
 * @type { MediaRecorder | null }
 */
let recorder;

/**
 * @type { Blob[] }
 */
let data = [];

// Function to start recording the media stream
async function startRecording(streamId) {
    if (recorder?.state === 'recording') {
        throw new Error('Called startRecording while recording is in progress.'); // Prevent multiple recordings
    }

    // Request media permissions to capture audio and video from the specified tab
    const media = await navigator.mediaDevices.getUserMedia({
        audio: {
            mandatory: {
                chromeMediaSource: 'tab',
                chromeMediaSourceId: streamId
            }
        },
        video: {
            mandatory: {
                chromeMediaSource: 'tab',
                chromeMediaSourceId: streamId,
                maxWidth: 1280,
                maxHeight: 720
            }
        }
    });

    // Connect the captured media to the audio output to continue playback
    const output = new AudioContext();
    const source = output.createMediaStreamSource(media);
    source.connect(output.destination);

    // Initialize the recorder with the captured media stream
    recorder = new MediaRecorder(media, { mimeType: 'video/webm' });
    recorder.ondataavailable = (event) => data.push(event.data); // Collect recorded data
    
    recorder.start(); // Start recording

    // Update the URL hash to indicate that recording is in progress
    // This allows the service worker to track the recording state via the URL
    window.location.hash = 'recording';
}

async function getSessionRecording() {
    await waitUntil(() => data.length > 0);
    const blob = new Blob(data, { type: 'video/webm' });
    return convertBlobToBase64(blob);
}

// Function to stop recording and clean up
async function stopRecording() {
    if (!recorder) {
        console.warn('Recorder is not initialized.'); // Warn if stop is called without an active recording
        return;
    }
    recorder.stop(); // Stop the recording
    recorder.stream.getTracks().forEach((t) => t.stop()); // Stop the media tracks
    
    // Remove the 'recording' state from the URL
    window.location.hash = '';
    
    const sessionRecording = await getSessionRecording();
    recorder = undefined;
    data = [];

    return sessionRecording
}
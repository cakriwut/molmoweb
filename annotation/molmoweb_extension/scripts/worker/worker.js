import debounce from "./debounce.js";
import waitUntil from "./waitUntil.js";

/**
 * @typedef RecordedInteractionEvent
 * @type {{ type: string; timestamp: number; [extraEventDataKey: string]: any, message?: string }}
 */

/**
 * @typedef RecordedInteractionEventWithExtraData
 * @type {{ event: RecordedInteractionEvent, screenshot?: string, html?: string, domSnapshot?: string, axTree: string }}
 */

/**
 * @typedef SendVideoEvent
 * @type {{ video: string | ArrayBuffer | null}}
 */

/**
 *
 * @param {number} tabId
 */
function injectEventCollectorScript(tabId) {
  chrome.scripting.executeScript({
    target: { tabId },
    files: ["scripts/content/event-collector.js"],
    injectImmediately: true,
  });
}

async function sendMessageToStartingTab(event) {
  const result = await chrome.storage.local.get(["startingTabId"]);
  const startingTabId = result.startingTabId;
  if (startingTabId) {
    return chrome.tabs.sendMessage(startingTabId, {
      data: event,
      type: "addEvent",
    });
  } else {
    console.error("startingTabId was not defined when an event was recorded");
  }
}

async function addRecordedEvent(event, tabId) {
  console.log("adding event");
  const result = await chrome.storage.local.get(["events"]);
  let events = result.events;
  if (!Array.isArray(events)) {
    events = [];
  }

  if (tabId) {
    event.event.tabId = tabId;
  }

  events.push(event);
  await chrome.storage.local.set({ events });

  try {
    await sendMessageToStartingTab(event);
  } catch (e) {
    console.error(
      "Something went wrong when sending an addEvent message to the starting tab",
      e,
    );
  }
}

// Detect when a tab's URL changes or the page reloads
async function tabUpdateListener(tabId, changeInfo, tab) {
  console.log("tabUpdateListener triggered");
  const result = await chrome.storage.local.get(["recordingWindowId"]);
  const recordingWindowId = result.recordingWindowId;
  const result2 = await chrome.storage.local.get(["recordingTabGroupId"]);
  const recordingTabGroupId = result2.recordingTabGroupId;
  if (tab.windowId === recordingWindowId) {
    if (changeInfo.status === "loading") {
      chrome.tabs.group({ groupId: recordingTabGroupId, tabIds: tab.id });
    }

    if (
      changeInfo.status === "loading" &&
      !tab.url?.includes("chrome://") &&
      !tab.pendingUrl?.includes("chrome://")
    ) {
      injectEventCollectorScript(tab.id);
    }
  }
}

// Detect when the user switches to another tab
async function tabSwitchListener(activeInfo) {
  const { tabId, windowId } = activeInfo;
  console.log("tabSwitchListener triggered:", { tabId, windowId });

  // Ensure the tab is in the recording window
  const result = await chrome.storage.local.get(["recordingWindowId"]);
  const recordingWindowId = result.recordingWindowId;

  if (windowId === recordingWindowId) {
    chrome.tabs.get(tabId, (tab) => {
      if (chrome.runtime.lastError) {
        console.error("Error getting tab info:", chrome.runtime.lastError);
        return;
      }

      console.log("Switched to tab:", tab);

      addRecordedEvent({
        event: {
          type: "tab-switched",
          tabId: tabId,
          url: tab.url || "unknown",
          title: tab.title || "unknown",
          timestamp: Date.now(),
        },
      });

      if (
        !tab.url?.includes("chrome://") &&
        !tab.url?.includes("about:blank")
      ) {
        injectEventCollectorScript(tabId);
      }
    });
  } else {
    const result = await chrome.storage.local.get(["recordingWindowId"]);
    const recordingWindowId = result.recordingWindowId;
    console.warn("Activated tab is not in the recording window:", {
      windowId,
      recordingWindowId,
    });
  }
}

// Detect when a new tab is created
function tabCreationListener(tab) {
  console.log("tabCreationListener triggered");

  const isExplicitUserAction =
    tab.pendingUrl === "chrome://newtab/" || // Explicit new tab (e.g., new tab button)
    (!tab.openerTabId && !tab.pendingUrl); // No opener, likely user-triggered

  addRecordedEvent({
    event: {
      type: "tab-created",
      tabId: tab.id,
      url: tab.url || "unknown",
      title: tab.title || "unknown",
      openedByAnotherTab: !isExplicitUserAction, // False if explicitly opened by the user
      timestamp: Date.now(),
    },
  });

  if (isExplicitUserAction) {
    console.log(`Tab ${tab.id} explicitly created by the user.`);
  } else {
    console.log(`Tab ${tab.id} created by a link or script.`);
  }
}

/**
 * @param {chrome.windows.Window} window
 */
async function handleWindowBoundsChanged(window) {
  // NOTE: This also includes moving a window around. If we don't want this we'll need to compare top/left to previous values.
  const result = await chrome.storage.local.get(["recordingWindowId"]);
  const recordingWindowId = result.recordingWindowId;
  if (window != null && window.id === recordingWindowId) {
    handleRecordInteraction(
      {
        type: "resizeWindow",
        height: window.height,
        width: window.width,
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight,
        timestamp: Date.now(),
      },
      null,
      false,
    );
  }
}

const debouncedHandleWindowBoundsChanged = debounce(
  handleWindowBoundsChanged,
  200,
);

function streamifyEvents(events) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode("["));
      let first = true;
      for (const event of events) {
        if (!first) {
          controller.enqueue(encoder.encode(","));
        }
        // Convert each event to JSON and encode it
        controller.enqueue(encoder.encode(JSON.stringify(event)));
        first = false;
      }
      controller.enqueue(encoder.encode("]"));
      controller.close();
    },
  });
}

async function compressEvents(events) {
  // Use the streaming approach instead of converting the entire array to a string first
  const eventStream = await streamifyEvents(events);
  const compressedStream = eventStream.pipeThrough(
    new CompressionStream("gzip"),
  );
  const response = new Response(compressedStream);
  return await response.blob();
}

/**
 * @param {Array<RecordedInteractionEvent | RecordedInteractionEventWithExtraData>} events
 * @param {string} uploadUrl
 * @param {string} sessionId
 */
async function uploadEvents(events, uploadUrl, sessionId) {
  const compressedEventsBlob = await compressEvents(events);

  const file = new File([compressedEventsBlob], sessionId + ".gz", {
    type: "application/gzip",
  });

  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(uploadUrl, {
    method: "POST",
    body: formData,
  });

  const responseText = await response.text();

  if (!response.ok) {
    throw new Error(`Upload error ${response.status}: ${responseText}`);
  }

  return responseText;
}

/**
 * @param {string} sessionRecording
 * @param {string} uploadUrl
 * @param {string} sessionId
 */
async function uploadVideo(sessionRecording, uploadUrl, sessionId) {
  const videoResponse = await fetch(sessionRecording);
  const videoBlob = await videoResponse.blob();

  const formData = new FormData();

  formData.append(
    "file",
    new File([videoBlob], sessionId + ".webm", { type: "video/webm" }),
  );

  const response = await fetch(uploadUrl, {
    method: "POST",
    body: formData,
  });

  const responseText = await response.text();

  if (!response.ok) {
    throw new Error(`Upload error ${response.status}: ${responseText}`);
  }

  return responseText;
}

/**
 *
 * @param {string | undefined} sessionRecording
 */
async function handleUpload(sessionRecording) {
  try {
    chrome.runtime.sendMessage({ type: "startUpload" });

    // This handles cases where the session ID and upload URL get wiped out. This lets us finish a session in case we don't have the right data available
    const result = await chrome.storage.local.get(["sessionId"]);
    const sessionId = result.sessionId;
    const sessionIdOrDefault = sessionId ?? crypto.randomUUID();
    const result2 = await chrome.storage.local.get(["uploadUrl"]);
    const uploadUrl = result2.uploadUrl;
    const uploadUrlOrDefault =
      uploadUrl ?? "http://127.0.0.1:5001/upload";

    const result3 = await chrome.storage.local.get(["events"]);
    const events = result3.events;
    const eventsUploadPromise = uploadEvents(
      events,
      uploadUrlOrDefault,
      sessionIdOrDefault,
    );

    if (sessionRecording) {
      uploadVideo(sessionRecording, uploadUrlOrDefault, sessionIdOrDefault);
    }

    const redirectLocation = await eventsUploadPromise;

    chrome.runtime.sendMessage({ type: "finishUpload" });

    chrome.runtime.sendMessage({ type: "finishSession", redirectLocation });
  } catch (e) {
    console.error("Something went wrong when uploading your session.", e);
    if (e instanceof Error) {
      chrome.runtime.sendMessage({
        type: "uploadFailed",
        detail: e.message,
        stack: e.stack,
        cause: e.cause,
      });
    } else {
      chrome.runtime.sendMessage({
        type: "uploadFailed",
        detail: JSON.stringify(e),
      });
    }

    throw e;
  }
}

async function handleSessionFinish() {
  try {
    // Stop recording and store the returned sessionRecording value in storage.
    const sessionRecordingResult = await chrome.runtime.sendMessage({
      type: "stop-recording",
      target: "offscreen",
    });
    await chrome.storage.local.set({
      sessionRecording: sessionRecordingResult,
    });

    // Retrieve events from storage.
    const eventsResult = await chrome.storage.local.get(["events"]);
    const events = eventsResult.events;
    console.log("Events on session finish:", events);

    // Wait until the last event is an 'unload' event.
    await waitUntil(() => events.at(-1)?.event.type === "unload", 3);

    // Retrieve the upload URL.
    const uploadResult = await chrome.storage.local.get(["uploadUrl"]);
    const uploadUrl = uploadResult.uploadUrl;
    console.log("Uploading files to", uploadUrl);

    // Retrieve sessionRecording from storage.
    const sessionResult = await chrome.storage.local.get(["sessionRecording"]);
    const sessionRecording = sessionResult.sessionRecording;
    await handleUpload(sessionRecording);

    // Notify content scripts.
    try {
      await sendMessageToStartingTab({ type: "finishSession" });
    } catch (e) {
      console.error("Error sending finishSession message to starting tab", e);
    }

    // Remove session-related keys.
    await chrome.storage.local.remove([
      "events",
      "sessionId",
      "startingTabId",
      "uploadUrl",
      "currentInstruction",
      "sessionRecording",
      "recordingWindowId",
      "recordingTabGroupId",
    ]);

    // Remove listeners.
    chrome.tabs.onUpdated.removeListener(tabUpdateListener);
    chrome.tabs.onCreated.removeListener(tabCreationListener);
    chrome.tabs.onActivated.removeListener(tabSwitchListener);
    chrome.windows.onBoundsChanged.removeListener(
      debouncedHandleWindowBoundsChanged,
    );
  } catch (e) {
    console.error("Something went wrong when finishing the session.", e);
  }
}

/**
 * Resizes a base64 encoded image (data URL) to fit within 1280x720 while maintaining the aspect ratio.
 * This version is suitable for a worker.
 * @param {string} dataURL - The original base64 data URL.
 * @returns {Promise<string>} - A promise that resolves to the resized image as a base64 data URL.
 */
function resizeImage(dataURL) {
  return fetch(dataURL)
    .then((response) => response.blob())
    .then((blob) => createImageBitmap(blob))
    .then((imageBitmap) => {
      const originalWidth = imageBitmap.width;
      const originalHeight = imageBitmap.height;
      // Compute scale factor
      const scaleFactor = Math.min(
        1280 / originalWidth,
        1280 / originalHeight,
        1,
      );
      const newWidth = originalWidth * scaleFactor;
      const newHeight = originalHeight * scaleFactor;

      // Create an OffscreenCanvas
      const canvas = new OffscreenCanvas(newWidth, newHeight);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(imageBitmap, 0, 0, newWidth, newHeight);

      // Convert canvas to a Blob
      return canvas.convertToBlob({ type: "image/png" });
    })
    .then((blob) => {
      // Convert the Blob to a base64 data URL using FileReader
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    });
}

async function recordWithScreenshot(payload) {
  const { event, html, takeScreenshot, tabId } = payload;
  const out = { event, html, screenshot: null };

  if (takeScreenshot) {
    try {
      const { recordingWindowId } =
        await chrome.storage.local.get("recordingWindowId");
      if (recordingWindowId != null) {
        // Send start loading message to side panel
        try {
          await chrome.runtime.sendMessage({
            type: "startScreenshotCapture",
          });
        } catch (e) {
          console.error(
            "Failed to send screenshot start message to side panel:",
            e,
          );
        }

        const shot = await maybeTakeScreenshot(recordingWindowId);
        if (shot) {
          out.screenshot = shot;
          // Send screenshot to side panel
          try {
            await chrome.runtime.sendMessage({
              type: "updateScreenshot",
              screenshot: shot,
            });
          } catch (e) {
            console.error("Failed to send screenshot to side panel:", e);
          }
        }
      }
    } catch (err) {
      console.error(
        "error capturing screenshot, falling back to raw event",
        err,
      );
    }
  } else {
    out.screenshot = null;
  }
  addRecordedEvent(out, tabId);
}

async function handleRecordInteraction(
  event,
  html = null,
  takeScreenshot = true,
  tabId = null,
) {
  await recordWithScreenshot({ event, html, takeScreenshot, tabId });
}

/**
 *
 * @param {string} sessionIdToStart
 * @param {string} instruction
 * @param {string} uploadUrlToStart
 * @param {number} tabId
 */
async function handleSessionStart(
  sessionIdToStart,
  instruction,
  task_steps,
  uploadUrlToStart,
  tabId,
) {
  const result = await chrome.storage.local.get(["sessionId"]);
  const sessionId = result.sessionId;
  console.log("sessionId:", sessionId);
  if (sessionId != null) {
    console.error("session already started");
  }

  console.log("start session", sessionIdToStart);
  console.log("instruction received", instruction);
  console.log("task_steps received", task_steps);
  console.log("upload URL", uploadUrlToStart);
  console.log("starting tab id", tabId);

  // Store the session-related values in chrome.storage.local.
  await chrome.storage.local.set({ ["events"]: [] });
  await chrome.storage.local.set({ ["sessionId"]: sessionIdToStart });
  await chrome.storage.local.set({ ["startingTabId"]: tabId });
  await chrome.storage.local.set({ ["uploadUrl"]: uploadUrlToStart });
  await chrome.storage.local.set({ ["currentInstruction"]: instruction });
  await chrome.storage.local.set({ ["currentTaskSteps"]: task_steps });

  const test = await chrome.storage.local.get(["startingTabId"]);
  console.log("Stored startingTabId:", test.startingTabId);

  // New window with default new tab (blank); participant navigates from the task instruction.
  const newWindow = await chrome.windows.create({
    focused: true,
    incognito: true,
  });
  await chrome.storage.local.set({ ["recordingWindowId"]: newWindow?.id });

  // Get all tabs in the new window and group them; store the group ID.
  const tabs = await chrome.tabs.query({ windowId: newWindow.id });
  const group = await chrome.tabs.group({
    createProperties: { windowId: newWindow.id },
    tabIds: tabs.map((tab) => tab.id),
  });
  await chrome.storage.local.set({ ["recordingTabGroupId"]: group });

  // Update the tab group appearance.
  chrome.tabGroups.update(group, { color: "red", title: "Task recording" });

  chrome.tabs.onUpdated.addListener(tabUpdateListener);
  chrome.tabs.onCreated.addListener(tabCreationListener);
  chrome.tabs.onActivated.addListener(tabSwitchListener);
  chrome.windows.onBoundsChanged.addListener(
    debouncedHandleWindowBoundsChanged,
  );
}

// when the user clicks the icon for the extension
chrome.action.onClicked.addListener(async (tab) => {
  if (chrome.sidePanel) {
    await chrome.sidePanel.open({ tabId: tab.id });
  }

  const existingContexts = await chrome.runtime.getContexts({});
  let recording = false;

  const offscreenDocument = existingContexts.find(
    (c) => c.contextType === "OFFSCREEN_DOCUMENT",
  );
  // If an offscreen document is not already open, create one.
  if (!offscreenDocument) {
    try {
      // Create an offscreen document.
      await chrome.offscreen.createDocument({
        url: "scripts/offscreen/offscreen.html",
        reasons: ["USER_MEDIA"],
        justification: "Recording from chrome.tabCapture API",
      });
    } catch (error) {
      console.error("Failed to create offscreen document:", error);
    }
  } else {
    recording = offscreenDocument.documentUrl.endsWith("#recording");
  }

  // don't record if we are already recording
  if (recording) {
    return;
  }
  // Get a MediaStream for the active tab.
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
  });

  // Send the stream ID to the offscreen document to start recording.
  chrome.runtime.sendMessage({
    type: "start-recording",
    target: "offscreen",
    data: streamId,
  });
});

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
  switch (message.type) {
    case "recordInteraction": {
      let event_type = message.event.type;
      let delay = 0;
      if (event_type === "click") {
        delay = 500;
      }
      setTimeout(async () => {
        await handleRecordInteraction(
          message.event,
          message.html,
          message.takeScreenshot,
          sender.tab.id,
        );
      }, delay);
      break;
    }

    case "sendNote": {
      await handleRecordInteraction(
        message.event,
        message.html,
        true,
        sender.tab?.id,
      );
      break;
    }

    case "sendQuestionAndAnswer": {
      await handleRecordInteraction(
        message.event,
        message.html,
        true,
        sender.tab.id,
      );
      break;
    }

    case "sendFinalAnswer": {
      console.log("Received sendFinalAnswer:", message);
      await handleRecordInteraction(
        message.event,
        message.html,
        true,
        sender.tab?.id,
      );

      const result = await chrome.storage.local.get(["sessionRecording"]);
      const sessionRecording = result.sessionRecording;
      await handleSessionFinish();
      break;
    }

    case "takeScreenshot": {
      console.log("Received takeScreenshot:", message);
      await handleRecordInteraction(
        message.event,
        message.html,
        true,
        sender.tab?.id,
      );
      break;
    }

    case "startSession": {
      await handleSessionStart(
        message.sessionId,
        message.instruction,
        message.task_steps,
        message.uploadUrl,
        sender.tab.id,
      );
      await chrome.storage.local.set({
        ["currentInstruction"]: message.instruction,
      });
      await chrome.storage.local.set({
        ["currentTaskSteps"]: message.task_steps,
      });
      console.log(message);
      console.log(message.task_steps);
      break;
    }

    case "getInstruction": {
      const result = await chrome.storage.local.get(["currentInstruction"]);
      const currentInstruction = result.currentInstruction;
      sendResponse({ instruction: currentInstruction });
      return true;
    }

    case "getWebsite": {
      sendResponse({ website: "" });
      return true;
    }

    case "retryUpload": {
      const result3 = await chrome.storage.local.get(["sessionRecording"]);
      const sessionRecording = result3.sessionRecording;
      await handleUpload(sessionRecording);
    }

    case "completeStep": {
      console.log("Received completeStep", message);
      console.log("Sender tab info:", sender.tab);

      // Get the active tab in the recording window instead of sender tab
      try {
        const { recordingWindowId } =
          await chrome.storage.local.get("recordingWindowId");
        if (recordingWindowId) {
          const tabs = await chrome.tabs.query({
            active: true,
            windowId: recordingWindowId,
          });
          const activeTab = tabs[0];
          console.log("Active tab in recording window:", activeTab);
          await handleRecordInteraction(
            message.event,
            message.html,
            true,
            activeTab?.id,
          );
        } else {
          console.log(
            "No recording window ID found, falling back to sender tab",
          );
          await handleRecordInteraction(
            message.event,
            message.html,
            true,
            sender.tab?.id,
          );
        }
      } catch (error) {
        console.error("Error getting active tab:", error);
        await handleRecordInteraction(
          message.event,
          message.html,
          true,
          sender.tab?.id,
        );
      }
      break;
    }

    case "completeTask": {
      console.log("Received completeTask:", message);
      await handleRecordInteraction(
        message.event,
        message.html,
        true,
        sender.tab?.id,
      );
      break;
    }

    default:
      console.warn("Unknown message type:", message.type);
  }
});

async function maybeTakeScreenshot(windowId) {
  let dataUrl = await chrome.tabs.captureVisibleTab(windowId, {
    format: "jpeg",
    quality: 50,
  });
  if (!dataUrl) return null;

  dataUrl = await resizeImage(dataUrl);
  return dataUrl;
}

chrome.commands.onCommand.addListener(async (command) => {
  console.log("Command received:", command);

  if (command === "take_screenshot") {
    console.log("Processing screenshot command...");

    try {
      // Get the active tab
      const [activeTab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });
      console.log("📋 Active tab:", activeTab);

      if (activeTab) {
        // Create a screenshot event similar to what the side panel does
        const screenshotEvent = {
          type: "takeScreenshot",
          note: "screenshot",
          timestamp: Date.now(),
        };

        console.log("Calling handleRecordInteraction...");
        // Handle the screenshot recording
        await handleRecordInteraction(
          screenshotEvent,
          null,
          true,
          activeTab.id,
        );
        console.log("Screenshot taken via keyboard shortcut");
      } else {
        console.error("No active tab found for screenshot command");
      }
    } catch (error) {
      console.error("Error in screenshot command:", error);
    }
  } else {
    console.log("Unknown command:", command);
  }
});

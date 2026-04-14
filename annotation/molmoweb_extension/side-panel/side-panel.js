async function sendMessageEvent(event) {
  try {
    await chrome.runtime.sendMessage({ type: event.type, event });
    console.log("Interaction recorded successfully:", event.type);
  } catch (error) {
    console.error("Failed to record interaction:", event.type, error);
  }

  return Promise.resolve();
}

(async function loadInstructionAndWebsite() {
  try {
    const { currentInstruction } = await chrome.storage.local.get([
      "currentInstruction",
    ]);
    document.getElementById("dynamic-instruction").textContent =
      currentInstruction || "No instruction found.";
  } catch (error) {
    console.error("Error retrieving instruction:", error);
    document.getElementById("dynamic-instruction").textContent =
      "No instruction found :().";
  }

  try {
    const { currentTaskSteps } = await chrome.storage.local.get([
      "currentTaskSteps",
    ]);
    console.log(currentTaskSteps);
    // Step Tracker UI
    const steps = currentTaskSteps;
    let completedSteps = 0;
    const totalSteps = steps.length;

    function renderStepTracker() {
      document.getElementById("dynamic-task-steps").innerHTML = `
          <div style="margin-bottom:12px;">
            <div style="background:#e2e8f0;height:6px;border-radius:3px;">
              <div id="progressFill" style="background:#10b981;height:100%;width:0%;transition:width 0.5s;border-radius:3px;"></div>
            </div>
          </div>
          <ul id="step-list" style="list-style:none;padding:0;margin:0;"></ul>
        `;

      const stepList = document.getElementById("step-list");
      stepList.innerHTML = steps
        .map(
          (step, i) => `
          <li class="step-item" data-step="${i}" style="display:flex;align-items:center;margin-bottom:10px;position:relative;">
            <div style="width:32px;height:32px;border-radius:50%;background:#e2e8f0;color:#64748b;display:flex;align-items:center;justify-content:center;font-weight:600;margin-right:12px;">${i + 1}</div>
            <div style="flex:1;">${step}</div>
            <button class="check-btn" style="width:32px;height:32px;border:2px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;margin-left:8px;" data-step-btn="${i}" data-step-text="${step}"><span>✓</span></button>
            <button class="note-btn" style="width:32px;height:32px;border:2px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;margin-left:8px;" data-note-btn="${i}" title="Can't complete this step?">?</button>
            <div class="note-input-container" data-note-input="${i}" style="display:none;position:absolute;top:48px;left:0;width:100%;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;z-index:2;">
              <textarea style="width:100%;border:1px solid #f87171;border-radius:6px;padding:8px;font-size:14px;resize:vertical;min-height:40px;font-family:inherit;" placeholder="Describe the issue you're encountering..."></textarea>
              <div style="margin-top:8px;display:flex;gap:8px;">
                <button class="btn-submit-note" data-submit-note="${i}" style="background:#ef4444;color:white;padding:6px 12px;border-radius:4px;font-size:12px;font-weight:500;cursor:pointer;border:none;">Report Issue</button>
                <button class="btn-cancel-note" data-cancel-note="${i}" style="background:#f3f4f6;color:#6b7280;padding:6px 12px;border-radius:4px;font-size:12px;font-weight:500;cursor:pointer;border:none;">Cancel</button>
              </div>
            </div>
          </li>
        `,
        )
        .join("");

      // Attach event listeners after rendering
      steps.forEach((step, i) => {
        const btn = document.querySelector(`button[data-step-btn="${i}"]`);
        if (btn) {
          btn.addEventListener("click", async function () {
            if (!btn.classList.contains("completed")) {
              btn.classList.add("completed");
              btn.style.background = "#10b981";
              btn.style.borderColor = "#10b981";
              btn.style.color = "#fff";
              completedSteps++;
              updateProgress();
              checkCompleteButton();

              let currentHtml = "";
              try {
                const [tab] = await chrome.tabs.query({
                  active: true,
                  currentWindow: true,
                });
                const results = await chrome.scripting.executeScript({
                  target: { tabId: tab.id },
                  function: () => document.documentElement.outerHTML,
                });
                currentHtml = results[0]?.result || "";
              } catch (error) {
                console.error("Failed to extract HTML:", error);
                currentHtml = "Failed to extract HTML";
              }

              const checkBtnEvent = {
                type: COMPLETE_STEP_EVENT,
                note: "complete a step",
                timestamp: Date.now(),
                taskTitle: step,
                html: currentHtml,
              };

              recordUserMessage(checkBtnEvent);
            }
          });
        }

        // Note button
        const noteBtn = document.querySelector(`button[data-note-btn="${i}"]`);
        if (noteBtn) {
          noteBtn.addEventListener("click", function () {
            // Remove any existing injected note section for other steps
            document
              .querySelectorAll(".dynamic-step-note-section")
              .forEach((el) => el.remove());

            // Inject Make a Note section below this step
            const stepItem = document.querySelector(
              `li.step-item[data-step="${i}"]`,
            );
            if (stepItem) {
              const noteSectionHTML = `
                  <div class="dynamic-step-note-section" style="margin-top:8px;">
                    <details open>
                      <summary>Make a Note
                        <span class="tooltip">?
                          <span class="tooltiptext">
                            Example 1:<br>
                            Task: Compare the cost of women's Vans classic slip-on skate shoes on 3 different websites.<br>
                            Note 1: the cost of the shoes on zappos.com is $45<br>
                            Note 2: the cost of the shoes on DSW is $69.99<br>
                            Note 3: the cost of the shoes on amazon.com is $24.99<br><br>
                            Example 2:<br>
                            Task: Navigate to alaskaairlines and find today's cheapest flight to Mexico City.<br>
                            Note: When I tried to navigate to alaskaairlines.com it detected me as a bot and blocked me.<br>
                          </span>
                        </span>
                      </summary>
                      <form class="action-form step-note-form">
                        <label>
                          <textarea name="note" class="textarea" required placeholder="Make a note here..."></textarea>
                        </label>
                        <button class="button green" type="submit">Send</button>
                      </form>
                    </details>
                  </div>
                `;
              stepItem.insertAdjacentHTML("afterend", noteSectionHTML);

              // Attach submit/cancel events
              const noteForm =
                stepItem.nextElementSibling.querySelector(".step-note-form");
              const cancelBtn = stepItem.nextElementSibling.querySelector(
                'button[type="button"]',
              );
              if (noteForm) {
                noteForm.addEventListener("submit", function (event) {
                  event.preventDefault();
                  const noteText = noteForm
                    .querySelector("textarea")
                    .value.trim();
                  if (noteText) {
                    sendMessageEvent({
                      type: "sendNote",
                      note: `${noteText}`,
                      step_id: i,
                      step_str: step,
                      timestamp: Date.now(),
                    });
                    recordUserMessage({
                      type: "sendNote",
                      note: `${noteText}`,
                      step_id: i,
                      step_str: step,
                      timestamp: Date.now(),
                    });
                    // Mark step visually as skipped
                    stepItem.style.opacity = "0.7";
                    const btn = stepItem.querySelector(".check-btn");
                    btn.classList.add("completed");
                    btn.style.background = "#f59e0b";
                    btn.style.borderColor = "#f59e0b";
                    btn.style.color = "#fff";
                    completedSteps++;
                    updateProgress();
                    checkCompleteButton();
                    // Remove note section
                    stepItem.nextElementSibling.remove();
                  }
                });
              }
              if (cancelBtn) {
                cancelBtn.addEventListener("click", function () {
                  stepItem.nextElementSibling.remove();
                });
              }
            }
          });
        }
      });

      updateProgress();
      checkCompleteButton();
    }

    function updateProgress() {
      const progressFill = document.getElementById("progressFill");
      const percentage = (completedSteps / totalSteps) * 100;
      if (progressFill) progressFill.style.width = percentage + "%";
    }

    function checkCompleteButton() {
      if (completedSteps === totalSteps) {
        sendMessageEvent({
          type: "completeTask",
          steps: steps,
          timestamp: Date.now(),
        });
        // Dynamically insert Finish Session section after step tracker
        if (!document.getElementById("final-answer-form")) {
          const finishSessionHTML = `
              <details open>
                <summary>Finish Session
                  <span class="tooltip">?
                    <span class="tooltiptext">
                      Example 1:<br>
                      If the instruction contained a question, your final answer would be an answer.<br><br>
                      Example 2:<br>
                      If the instruction did not explicitly contain a question, your final answer will be a conclusion.<br><br>
                      Example 3:<br>
                      If you were blocked by the task requiring personal or payment information, your final answer would explain the obstacle.<br><br>
                      Example 4:<br>
                      If the instruction was completely infeasible, you may send a final answer that says the task was infeasible and why.<br>
                    </span>
                  </span>
                </summary>
                <form id="final-answer-form" class="action-form">
                  <label>
                    <textarea name="answer" class="textarea" required placeholder="Type your final answer here..."></textarea>
                  </label>
                  <button class="button pink" type="submit">
                    Send and Finish
                  </button>
                  <p id="final-answer-form-loading-indicator" data-hidden="true" style="font-weight: bold;">Uploading...</p>
                  <p>Please wait a few moments after pressing Send and Finish.</p>
                </form>
              </details>
            `;
          const dynamicTaskSteps =
            document.getElementById("dynamic-task-steps");
          dynamicTaskSteps.insertAdjacentHTML("afterend", finishSessionHTML);

          // Attach JS event for final-answer-form
          const finalAnswerForm = document.getElementById("final-answer-form");
          if (finalAnswerForm) {
            finalAnswerForm.addEventListener("submit", function (event) {
              event.preventDefault();
              const formData = new FormData(finalAnswerForm);
              const finalAnswerEvent = {
                type: "sendFinalAnswer",
                answer: formData.get("answer"),
                timestamp: Date.now(),
              };
              sendMessageEvent(finalAnswerEvent);
              finalAnswerForm.reset();
              document
                .getElementById("final-answer-form-loading-indicator")
                .removeAttribute("data-hidden");
              setTimeout(() => {
                document
                  .getElementById("final-answer-form-loading-indicator")
                  .setAttribute("data-hidden", "true");
              }, 1500);
            });
          }
        }
        // Optionally scroll to the new section
        setTimeout(() => {
          const finalForm = document.getElementById("final-answer-form");
          if (finalForm) finalForm.scrollIntoView({ behavior: "smooth" });
        }, 100);
      }
    }

    // Render after DOM loaded
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderStepTracker);
    } else {
      renderStepTracker();
    }
  } catch (error) {
    console.error("Error retrieving task steps:", error);
    document.getElementById("dynamic-task-steps").textContent =
      "No task steps found :(";
  }

})();

const SEND_NOTE_EVENT = "sendNote";
const SEND_FINAL_ANSWER_EVENT = "sendFinalAnswer";
const SEND_QUESTION_AND_ANSWER_EVENT = "sendQuestionAndAnswer";
const TAKE_SCREENSHOT_EVENT = "takeScreenshot";
const COMPLETE_STEP_EVENT = "completeStep";

/**
 * @typedef UserMessageEvent
 * @type {SEND_NOTE_EVENT | SEND_FINAL_ANSWER_EVENT | SEND_QUESTION_AND_ANSWER_EVENT | TAKE_SCREENSHOT_EVENT | COMPLETE_STEP_EVENT}
 *
 * @typedef {object} NoteEvent
 * @property {SEND_NOTE_EVENT} type
 * @property {string} note
 *
 * @typedef FinalAnswerEvent
 * @property {SEND_FINAL_ANSWER_EVENT} type
 * @property {string} answer
 *
 * @typedef QuestionAndAnswerEvent
 * @property {SEND_QUESTION_AND_ANSWER_EVENT} type
 * @property {string} question
 * @property {string} answer
 *
 * @typedef ScreenshotEvent
 * @property {TAKE_SCREENSHOT_EVENT} type
 * @property {string} note
 *
 * @typedef CompleteStepEvent
 * @property {COMPLETE_STEP_EVENT} type
 * @property {string} note
 * @property {string} title
 */

/**
 *
 * @type {Record<UserMessageEvent, string>} eventType
 */
const eventLabelMap = {
  [SEND_NOTE_EVENT]: "Note",
  [SEND_FINAL_ANSWER_EVENT]: "Final answer",
  [SEND_QUESTION_AND_ANSWER_EVENT]: "Question & answer",
  [TAKE_SCREENSHOT_EVENT]: "Screenshot",
  [COMPLETE_STEP_EVENT]: "Complete Step",
};

/**
 *
 * @param {NoteEvent | FinalAnswerEvent | QuestionAndAnswerEvent | ScreenshotEvent | CompleteStepEvent} event
 */
function createMessageRow(event) {
  const messageElement = document.createElement("li");

  const titleElement = document.createElement("strong");
  titleElement.innerHTML = eventLabelMap[event.type];
  messageElement.appendChild(titleElement);

  Object.entries(event)
    .filter(([key, _]) => key !== "type")
    .forEach(([key, value]) => {
      if (key === "axTree") return;
      if (key === "html") return;
      const containerElement = document.createElement("div");
      const valueElement = document.createElement("span");
      valueElement.innerHTML = `${key}: ${value}`;

      containerElement.appendChild(valueElement);
      messageElement.appendChild(containerElement);
    });

  document.getElementById("user-messages-list")?.appendChild(messageElement);
}

/**
 *
 * @param {NoteEvent | FinalAnswerEvent | QuestionAndAnswerEvent | ScreenshotEvent | CompleteStepEvent} event
 * @returns
 */
function recordUserMessage(event) {
  sendMessageEvent(event).then((_) => {
    createMessageRow(event);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  // Add Enter key event listeners for all textareas
  function addEnterKeyListener(formId, textareaSelector) {
    const form = document.getElementById(formId);
    const textarea = form?.querySelector(textareaSelector);

    if (textarea) {
      textarea.addEventListener("keydown", (event) => {
        // Submit on Enter, but allow Shift+Enter for new lines
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          form.dispatchEvent(
            new Event("submit", { cancelable: true, bubbles: true }),
          );
        }
      });
    }
  }

  // Add Enter key listeners to all forms
  addEnterKeyListener("note-form", 'textarea[name="note"]');
  addEnterKeyListener("final-answer-form", 'textarea[name="answer"]');
  addEnterKeyListener("question-answer-form", 'textarea[name="question"]');
  addEnterKeyListener("question-answer-form", 'textarea[name="answer"]');

  document
    .getElementById("question-answer-form")
    ?.addEventListener("submit", (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);

      const questionAndAnswerEvent = {
        type: SEND_QUESTION_AND_ANSWER_EVENT,
        question: formData.get("question"),
        answer: formData.get("answer"),
        timestamp: Date.now(),
      };

      recordUserMessage(questionAndAnswerEvent);
      event.target?.reset();
    });

  document.getElementById("note-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);

    const noteEvent = {
      type: SEND_NOTE_EVENT,
      note: formData.get("note"),
      timestamp: Date.now(),
    };

    recordUserMessage(noteEvent);
    event.target?.reset();
  });

  document
    .getElementById("final-answer-form")
    ?.addEventListener("submit", (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);

      const finalAnswerEvent = {
        type: SEND_FINAL_ANSWER_EVENT,
        answer: formData.get("answer"),
        timestamp: Date.now(),
      };

      recordUserMessage(finalAnswerEvent);
      event.target?.reset();
    });

  document
    .getElementById("retry-upload-button")
    ?.addEventListener("click", () => {
      chrome.runtime.sendMessage({ type: "retryUpload" });
    });

  document
    .getElementById("screenshot-button")
    ?.addEventListener("click", () => {
      // Show loading spinner immediately when button is clicked
      showScreenshotLoading();

      const screenshotEvent = {
        type: TAKE_SCREENSHOT_EVENT,
        note: "screenshot",
        timestamp: Date.now(),
      };

      recordUserMessage(screenshotEvent);
    });
});

/**
 *
 * @param {string} redirectUrl
 */
function handleSessionFinish(redirectUrl) {
  document
    .getElementById("instructions-and-answers")
    ?.setAttribute("data-hidden", "true");

  const redirectElement = document.getElementById("redirect-location");
  const sessionInstructions = document.getElementById(
    "session-finished-instructions",
  );

  if (redirectUrl && redirectUrl.trim() !== "") {
    redirectElement.setAttribute("href", redirectUrl);
    redirectElement.textContent = redirectUrl;
    sessionInstructions.innerHTML = `<p>Thank you for recording your session. Please go to <a href="${redirectUrl}" target="_blank">this link</a> to ensure your contribution has been recorded.</p>`;
  } else {
    redirectElement.removeAttribute("href");
    redirectElement.textContent = "";
    sessionInstructions.innerHTML = `<p>Thank you for recording your session. Your session has been completed successfully.</p>`;
  }

  sessionInstructions?.removeAttribute("data-hidden");
}

function handleUploadFailed(event) {
  document
    .getElementById("instructions-and-answers")
    ?.setAttribute("data-hidden", "true");
  document
    .getElementById("final-answer-form-loading-indicator")
    ?.setAttribute("data-hidden", "true");
  document
    .getElementById("upload-failed-instructions")
    ?.setAttribute("data-hidden", "false");
  const uploadFailedDetailsElement = document.getElementById(
    "upload-failed-details",
  );

  if (!uploadFailedDetailsElement) {
    return;
  }

  uploadFailedDetailsElement.innerHTML = Object.entries(event).reduce(
    (acc, [key, value]) => {
      if (key === "type") {
        return acc;
      }

      return acc + `<strong>${key}</strong>: <code>${value}</code><br><br>`;
    },
    "",
  );
}

function handleStartLoading() {
  document
    .getElementById("upload-failed-instructions")
    ?.setAttribute("data-hidden", "true");
  document
    .querySelectorAll('.action-form button[type="submit"]')
    .forEach((element) => {
      element.setAttribute("disabled", "true");
    });

  document
    .getElementById("final-answer-form-loading-indicator")
    ?.removeAttribute("data-hidden");
}

function handleFinishLoading() {
  document
    .querySelectorAll('.action-form button[type="submit"]')
    .forEach((element) => {
      element.removeAttribute("disabled");
    });
  document
    .getElementById("final-answer-form-loading-indicator")
    ?.setAttribute("data-hidden", "true");
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  switch (message.type) {
    case "finishSession":
      console.log("finishing session");
      handleSessionFinish(message.redirectLocation);
      break;

    case "startUpload":
      handleStartLoading();
      break;

    case "finishUpload":
      handleFinishLoading();
      break;

    case "uploadFailed":
      handleUploadFailed(message);
      break;

    case "updateScreenshot":
      updateScreenshotDisplay(message.screenshot);
      break;

    case "startScreenshotCapture":
      showScreenshotLoading();
      break;
  }
});

/**
 * Updates the screenshot display in the side panel
 * @param {string} screenshotDataUrl - The screenshot as a data URL
 */
function updateScreenshotDisplay(screenshotDataUrl) {
  const screenshotImg = document.getElementById("latest-screenshot");
  const noScreenshotMessage = document.getElementById("no-screenshot-message");
  const loadingSpinner = document.getElementById("screenshot-loading-spinner");

  if (screenshotImg && noScreenshotMessage && loadingSpinner) {
    // Hide loading spinner
    loadingSpinner.style.display = "none";

    screenshotImg.src = screenshotDataUrl;
    screenshotImg.style.display = "block";
    noScreenshotMessage.style.display = "none";
  }
}

/**
 * Shows the screenshot loading spinner
 */
function showScreenshotLoading() {
  const screenshotImg = document.getElementById("latest-screenshot");
  const noScreenshotMessage = document.getElementById("no-screenshot-message");
  const loadingSpinner = document.getElementById("screenshot-loading-spinner");

  if (screenshotImg && noScreenshotMessage && loadingSpinner) {
    // Hide existing content and show spinner
    screenshotImg.style.display = "none";
    noScreenshotMessage.style.display = "none";
    loadingSpinner.style.display = "flex";
  }
}

/**
 * Extracts data items from ARIA attributes (browsergym format)
 * @param {string} ariaValue - The ARIA attribute value
 * @returns {Array} - Array containing extracted data and new value
 */
function extractDataItemsFromAria(ariaValue) {
  const dataPattern = /browsergym_id:(\w+)/;
  const match = ariaValue.match(dataPattern);
  if (match) {
    const browsergymId = match[1];
    const newValue = ariaValue.replace(dataPattern, "").trim();
    return [[browsergymId], newValue];
  }
  return [[], ariaValue];
}

/**
 * Simple test function to verify debugger API works
 * @param {number} tabId - The tab ID to test
 * @returns {Promise<boolean>} - Whether debugger attachment works
 */
async function testDebuggerAccess(tabId) {
  return new Promise((resolve) => {
    console.log("Testing debugger access for tabId:", tabId);

    chrome.debugger.attach({ tabId }, "1.3", () => {
      if (chrome.runtime.lastError) {
        console.error("Debugger attach test FAILED:", chrome.runtime.lastError);
        // Check for specific Chrome URL error
        if (
          chrome.runtime.lastError.message &&
          chrome.runtime.lastError.message.includes("chrome://")
        ) {
          console.warn("Cannot attach debugger to Chrome internal page");
        }
        resolve(false);
        return;
      }

      console.log("Debugger attach test SUCCESSFUL");
      chrome.debugger.detach({ tabId }, () => {
        console.log("Debugger detached after test");
        resolve(true);
      });
    });
  });
}

/**
 * Extracts the AXTree of all frames using Chrome DevTools Protocol
 * @param {number} tabId - The tab ID to extract AXTree from
 * @returns {Promise<Object>} - Dictionary of AXTrees indexed by frame IDs
 */
async function extractAllFrameAxtrees(tabId) {
  console.log("Starting AXTree extraction for tabId:", tabId);

  // First test if debugger access works
  const debuggerWorks = await testDebuggerAccess(tabId);
  if (!debuggerWorks) {
    throw new Error("Debugger access test failed");
  }

  return new Promise((resolve, reject) => {
    // Attach debugger to the tab
    chrome.debugger.attach({ tabId }, "1.3", () => {
      if (chrome.runtime.lastError) {
        console.error("Failed to attach debugger:", chrome.runtime.lastError);
        reject(chrome.runtime.lastError);
        return;
      }

      console.log("Debugger attached successfully");

      // Enable Accessibility domain
      chrome.debugger.sendCommand({ tabId }, "Accessibility.enable", {}, () => {
        if (chrome.runtime.lastError) {
          console.error(
            "Failed to enable Accessibility:",
            chrome.runtime.lastError,
          );
          chrome.debugger.detach({ tabId });
          reject(chrome.runtime.lastError);
          return;
        }

        console.log("Accessibility domain enabled");

        // Get frame tree
        chrome.debugger.sendCommand(
          { tabId },
          "Page.getFrameTree",
          {},
          (frameTreeResult) => {
            if (chrome.runtime.lastError) {
              console.error(
                "Failed to get frame tree:",
                chrome.runtime.lastError,
              );
              chrome.debugger.detach({ tabId });
              reject(chrome.runtime.lastError);
              return;
            }

            console.log("Frame tree retrieved:", frameTreeResult);

            // Extract all frame IDs (breadth-first search)
            const frameIds = [];
            const rootFrame = frameTreeResult.frameTree;
            const framesToProcess = [rootFrame];

            while (framesToProcess.length > 0) {
              const frame = framesToProcess.pop();
              framesToProcess.push(...(frame.childFrames || []));
              frameIds.push(frame.frame.id);
            }

            console.log("Found frame IDs:", frameIds);

            // Extract AXTree for each frame
            const frameAxtrees = {};
            let completedFrames = 0;

            if (frameIds.length === 0) {
              console.warn("No frames found");
              chrome.debugger.detach({ tabId });
              resolve({});
              return;
            }

            frameIds.forEach((frameId) => {
              chrome.debugger.sendCommand(
                { tabId },
                "Accessibility.getFullAXTree",
                { frameId },
                (axTreeResult) => {
                  if (chrome.runtime.lastError) {
                    console.error(
                      `Failed to get AXTree for frame ${frameId}:`,
                      chrome.runtime.lastError,
                    );
                  } else {
                    console.log(
                      `Got AXTree for frame ${frameId}, nodes:`,
                      axTreeResult?.nodes?.length || 0,
                    );
                    frameAxtrees[frameId] = axTreeResult;

                    if (axTreeResult && axTreeResult.nodes) {
                      axTreeResult.nodes.forEach((node) => {
                        let dataItems = [];

                        // Look for data in node's "roledescription" property
                        if (node.properties) {
                          for (let i = 0; i < node.properties.length; i++) {
                            const prop = node.properties[i];
                            if (
                              prop.name === "roledescription" &&
                              prop.value &&
                              prop.value.value
                            ) {
                              const [extractedData, newValue] =
                                extractDataItemsFromAria(prop.value.value);
                              dataItems = extractedData;
                              prop.value.value = newValue;
                              // Remove property if empty
                              if (newValue === "") {
                                node.properties.splice(i, 1);
                              }
                              break;
                            }
                          }
                        }

                        // Look for data in node's "description" (fallback)
                        if (node.description && node.description.value) {
                          const [extractedDataBis, newValue] =
                            extractDataItemsFromAria(node.description.value);
                          node.description.value = newValue;
                          if (newValue === "") {
                            delete node.description;
                          }
                          if (!dataItems.length) {
                            dataItems = extractedDataBis;
                          }
                        }

                        if (dataItems.length > 0) {
                          const [dataItemId] = dataItems;
                          node.browsergym_id = dataItemId;
                        }
                      });
                    }
                  }

                  completedFrames++;
                  console.log(
                    `Completed ${completedFrames}/${frameIds.length} frames`,
                  );

                  if (completedFrames === frameIds.length) {
                    // Detach debugger and return results
                    chrome.debugger.detach({ tabId }, () => {
                      console.log(
                        "AXTree extraction complete, returning:",
                        Object.keys(frameAxtrees),
                      );
                      resolve(frameAxtrees);
                    });
                  }
                },
              );
            });
          },
        );
      });
    });
  });
}

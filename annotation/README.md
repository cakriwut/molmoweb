# Browser annotation (FastHTML server + Chrome extension)

This folder contains a minimal, self-hosted stack for running web-browsing annotation sessions:

- **`browser_app.py`** — FastHTML app that serves task pages, receives uploads at `/upload`, and stores files on **disk** under a configurable directory (default `./uploads`).
- **`molmoweb_extension/`** — Chrome extension that records interaction data in an incognito session and POSTs files to the annotation server.


## Prerequisites

- Python 3.11+ (3.12+ recommended; f-string parsing matches the upstream app).
- Google Chrome.

## 1. Task configuration

Copy `configs/example_tasks.json` and edit the `tasks` map. Each task id (the JSON object key, e.g. `demo`) is what participants select on the home page. Per task, the following four keys are required:

- `domain` — Short label shown in the task details table.
- `instruction` — Full instruction string for the task page and extension side panel.
- `task_name` — Display name in the task details table.
- `task_steps` — List of step strings (used by the extension payload).

When participants click **Start Session**, the extension opens a new window on a blank new tab. Put any starting URL or navigation guidance in the instruction text (and task steps) so they know where to go.

Optional top-level keys:

| Key | Purpose |
| --- | --- |
| `study_title` | Shown on the home page. |
| `stid` | Optional global study id used as a directory prefix (otherwise per-session `study_id` is used). |

## 2. Environment variables

cp `.env.example` -> `.env` and update the env values. 

- **`ANNOTATION_DATA_DIR`** — Absolute or relative path where uploads are written. Default: `uploads/` next to `browser_app.py`.
- **`ANNOTATION_CONFIG`** — Path to your tasks JSON (instead of `--config`).
- **`ANNOTATION_PORT`** — Listen port (default `5001`).
- **`ENV=development`** — Enables auto-reload while editing `browser_app.py`.

## 3. Chrome extension

### Load unpacked (development)

1. Open `chrome://extensions`, enable **Developer mode**, **Load unpacked**, and select the `molmoweb_extension` directory.
2. Open **Details** for the extension and enable **Allow in Incognito**. 
  -- We recommend keeping `incognito: true` in the extension when using this tool at scale so that personal information is not recorded. If you wish to not use incognito windows, change `incognito: false` in the extension `worker.js`.
3. The stock `manifest.json` only injects the session starter on `http://localhost:5001/*` and `http://127.0.0.1:5001/*`. For another host or port, add a match pattern under `content_scripts[0].matches`, for example `"https://your-domain.example/*"`, then click **Reload** on the extension.

## 4. Install and run the server

```bash
cd molmoweb-internal/annotation
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
python browser_app.py --config configs/example_tasks.json
```

Open `http://127.0.0.1:5001/` (or your chosen port), pick a task id, and follow the on-page steps.

Watch this quick video to see how the browser app and Chrome extension work: [annotation tool demo](./annotation_tool_demo.mp4). **It is very important that you open the chrome extension side panel immediately after the session is started, otherwise browsing data will not be saved correctly.**

## 5. Upload directory layout

For each finished session, the server writes (under `ANNOTATION_DATA_DIR`):

- `{study_id}/{task_id}.gz` — compressed event stream  
- `{study_id}/{task_id}.webm` — screen recording (when present)  
- `configs/{study_id}/{task_id}.json` — session metadata snapshot  

`uploads/` is listed in `.gitignore` so local runs do not commit participant data.


### Upload URL behavior

When the task page starts a session, it passes `uploadUrl: window.location.origin + "/upload"` into the extension. If storage is missing a saved URL, the service worker falls back to `http://127.0.0.1:5001/upload` for local debugging only. For real deployments, participants should complete flows from your deployed task page so the correct origin is used.


## Repository layout

```
annotation/
  browser_app.py          # FastHTML app entrypoint
  requirements.txt
  configs/example_tasks.json  # Directory contain sample tasks config json
  annotation_tool_demo.mp4  # Quick app + extension walkthrough
  uploads/                # Created at runtime (default data dir; gitignored)
  molmoweb_extension/     # Chrome extension (load unpacked)
```

## Optional: syncing to S3 or another backend

If you want object storage later, point a cron job or `aws s3 sync` (or rclone) at `ANNOTATION_DATA_DIR` instead of adding AWS to this process.

import argparse
import asyncio
import json
import logging
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, cast

from fasthtml.common import *

IS_DEVELOPMENT_ENVIRONMENT = (
    os.environ.get("ENV", "production") == "development"
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.environ.get("ANNOTATION_DATA_DIR", str(APP_DIR / "uploads"))
).expanduser().resolve()


def _path_under_data_dir(relative_key: str) -> Path:
    rel = Path(relative_key)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Invalid upload path")
    base = DATA_DIR.resolve()
    out = (base / rel).resolve()
    out.relative_to(base)
    return out


def _copy_upload_to_file(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with dest.open("wb") as out:
        shutil.copyfileobj(upload.file, out)

# Create the FastHTML app with specific settings
app, rt, actions, Action = fast_app(
    "actions.db",
    curr_category=str,
    curr_instruction_idx=int,
    feedback=str,
    debug=False,
    live=IS_DEVELOPMENT_ENVIRONMENT,
    hdrs=(
        picolink,
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.colors.min.css",
            type="text/css",
        ),
        Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css",
            type="text/css",
        ),
    ),
)


parser = argparse.ArgumentParser(
    description="Browser annotation server (FastHTML + Chrome extension)."
)
parser.add_argument(
    "--config",
    default=os.environ.get("ANNOTATION_CONFIG", ""),
    help="Path to task JSON (see configs/example_tasks.json). "
    "Or set ANNOTATION_CONFIG.",
)
parser.add_argument(
    "--port",
    type=int,
    default=int(os.environ.get("ANNOTATION_PORT", "5001")),
    help="Listen port (default: 5001 or ANNOTATION_PORT).",
)
args = parser.parse_args()

if not (args.config or "").strip():
    raise SystemExit(
        "Missing tasks config. Pass --config /path/to/tasks.json "
        "or set ANNOTATION_CONFIG."
    )
config_path = Path(args.config).expanduser().resolve()
if not config_path.is_file():
    raise SystemExit(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)


_TASK_REQUIRED_KEYS = ("domain", "instruction", "task_name", "task_steps")


def _load_task_configs(raw_tasks: dict) -> dict:
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise SystemExit('Config must include a non-empty "tasks" object.')
    out = {}
    for tid, entry in raw_tasks.items():
        if not isinstance(entry, dict):
            raise SystemExit(f'Task "{tid}": value must be a JSON object.')
        missing = [k for k in _TASK_REQUIRED_KEYS if k not in entry]
        if missing:
            raise SystemExit(
                f'Task "{tid}": missing required keys: {", ".join(missing)}. '
                f"See configs/example_tasks.json."
            )
        if not isinstance(entry["task_steps"], list):
            raise SystemExit(f'Task "{tid}": "task_steps" must be a JSON array.')
        out[tid] = dict(entry)
    return out


STUDY_TITLE = config.get("study_title", "Web browsing annotation")

global_stid = None
if "stid" in config:
    global_stid = config["stid"]
task_configs = _load_task_configs(config.get("tasks") or {})
session_configs = dict()


@app.get("/")
def index():
    return Titled(
        STUDY_TITLE,
        P("Available task IDs"),
        Ul(*[Li(A(f"Task ID {tid}", href=f"/{tid}")) for tid in task_configs]),
    )


def make_task_selector(
    sid: str, curr_instruction_idx: int = 0, curr_category: Optional[str] = None
):
    curr_category = session_configs[sid]["domain"]
    curr_desc = session_configs[sid]["instruction"]
    domain = session_configs[sid]["domain"]
    task_name = session_configs[sid]["task_name"]

    return Div(
        Card(
            H3("Task Details"),
            Table(
                Tr(Td(B("Domain")), Td(domain)),
                Tr(Td(B("Task Name")), Td(task_name)),
                Tr(Td(B("Instruction")), Td(curr_desc)),
                cls="striped",
            ),
            id="selected_instruction",
        ),
        id="taskSelector",
    )


def make_uploader(sid):
    return Div(
        Div(
            Br(),
            B("Recorded Data"),
            P("You may (optionally) view the recorded data below:"),
            Table(
                Tbody(
                    id="recording-contents",
                    hx_trigger="addEvent",
                    hx_post=create_recorded_event_row,
                    hx_swap="beforeend",
                    hx_vals="js:{event: event.detail?.event, screenshot: event.detail?.screenshot}",
                ),
                cls="striped",
                style="width:100%;",
            ),
            id="recorded-data-container",
        ),
        id="uploader",
    )


def FileMetaDataCard(msg="", content=None):
    if content is None:
        return Card(msg)
    return Article(
        Header(msg),
        content if content else "",
    )


@rt
def create_recorded_event_row(event: str, screenshot: Optional[str]):
    # IDK how to get htmx to send undefined as undefined so we're checking for the string "undefined" here
    if screenshot is None or len(screenshot) == 0 or screenshot == "undefined":
        img = ""
    else:
        img = Img(src=screenshot, style="max-width:100%; height:auto;")

    try:
        json_data = json.loads(event)
        json_str = json.dumps(json_data, indent=2)
    except:
        return ""

    # Render different row templates based on type of event
    event_type = json_data.get("type")
    event_timestamp = json_data.get("timestamp")
    event_video = json_data.get("video")

    # message template
    if event_type in {"sendFinalAnswer", "sendNote", "sendQuestionAndAnswer"}:
        dt = datetime.fromtimestamp(event_timestamp / 1000, timezone.utc)
        formatted_time = dt.strftime("%b %d, %Y %I:%M %p")

        if event_type == "sendFinalAnswer":
            border_color = "rgb(240, 82, 156)"
            label = "Final Answer"
            children = Div(f"Answer: {json_data.get("answer")}")
        elif event_type == "sendNote":
            border_color = "rgba(15, 203, 140, 1)"
            label = "Note"
            children = Div(f"Note: {json_data.get("note")}")
        elif event_type == "sendQuestionAndAnswer":
            border_color = "rgba(15, 203, 140, 1)"
            label = "Question & answer"
            children = Div(
                Div(f"Question: {json_data.get("question")}"),
                Div(f"Answer: {json_data.get("answer")}"),
            )

        return Tr(
            Td(
                Fieldset(
                    Legend(label, style="padding: 8px"),
                    Blockquote(formatted_time, children, style="margin: 0;"),
                    style=f"border: 1px solid {border_color};",
                ),
                style="width: 50%;",
            ),
            Td(img, style="width: 50%;"),
            style="display: flex;",
        )

    # video template
    if event_type == "send_video" and event_video is not None:
        return Tr(
            Td(
                Video(
                    Source(src=event_video, type="video/mp4"),
                    width="100%",
                    controls=True,
                ),
                style="width: 100%; margin-top: 10px;",
            ),
            style="display: flex;",
        )

    # all other events template
    return Tr(
        Td(
            Pre(Code(json_str)),
            style="width: 50%; white-space: pre-wrap; word-wrap: break-word;",
        ),
        Td(img, style="width: 50%;"),
        style="display: flex;",
    )


@rt
async def upload(request: Request):
    # multiple file upload taken from https://www.danielcorin.com/til/fasthtml/upload-multiple-images/
    form = await request.form()
    files_to_upload = cast(list[UploadFile], form.getlist("file"))
    global session_configs

    if not files_to_upload:
        return FileMetaDataCard("No files received")

    logging.getLogger("uvicorn").info(files_to_upload)

    try:
        config = None
        for file in files_to_upload:
            sid, file_extension = os.path.splitext(file.filename or "")
            config = session_configs.get(sid)
            stid_from_config = (
                config.get("study_id", "default_study")
                if config is not None
                else "default_study"
            )
            stid = (
                global_stid if global_stid is not None else stid_from_config
            )
            filename = (
                config.get("task_id", sid) if config is not None else sid
            )
            file_key = f"{stid}/{filename}{file_extension}"
            dest = _path_under_data_dir(file_key)
            await asyncio.to_thread(_copy_upload_to_file, file, dest)

        file_key = file_key.replace(".webm", ".json").replace(".gz", ".json")
        logging.getLogger("uvicorn").info("Saved session files under %s", file_key)
        cfg_path = _path_under_data_dir(f"configs/{file_key}")

        def _write_config_snapshot() -> None:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

        await asyncio.to_thread(_write_config_snapshot)

        return ""
    except ValueError as e:
        msg = str(e)
    except Exception as e:
        logging.getLogger("uvicorn").error(
            "An error occurred when uploading", exc_info=True, stack_info=True
        )
        msg = f"An error occurred: {str(e)}"

    return FileMetaDataCard(msg)


def make_steps(sid: str, num_steps: int):
    instruction = session_configs[sid]["instruction"]
    task_steps = session_configs[sid]["task_steps"]

    steps = [
        Div(
            Input(
                B("Step 1: "),
                "You will be completing a web browsing task specified in the instruction below. ",
                Br(),
                Br(),
                make_task_selector(
                    sid,
                    session_configs[sid].get("curr_instruction_idx", 0),
                    session_configs[sid].get("curr_category", None),
                ),
                type="checkbox",
                name="step",
                value=1,
                hx_post=f"/{sid}/steps",
                hx_target="#steps",
                hx_swap="outerHTML",
                hx_trigger="change",
                checked=num_steps > 1,
            )
        ),
        Br(),
        Div(
            Input(
                B("Step 2: "),
                "If you have installed the ",
                Code("chrome extension"),
                ", proceed to the next step.",
                Br(),
                I(
                    "If you haven't installed and enabled the extension, do so and refresh the page."
                ),
                type="checkbox",
                name="step",
                value=2,
                hx_post=f"/{sid}/steps",
                hx_target="#steps",
                hx_swap="outerHTML",
                hx_trigger="change",
                checked=num_steps > 2,
            )
        ),
        Br(),
        Div(
            Input(
                B("Step 3: "),
                "Click the ",
                Code("Start Session"),
                " button below to open a new incognito window and perform the task as per the task instruction.",
                "",
                type="checkbox",
                name="step",
                value=3,
                disabled=True,  # Disable the checkbox by default
                hx_post=f"/{sid}/steps",
                hx_target="#steps",
                hx_swap="outerHTML",
                hx_trigger="change",
                checked=num_steps > 3,
            ),
            Br(),
            Br(),
            (
                (
                    Div(
                        Button(
                            "Start Session",
                            cls="secondary col-xs-4",
                            style="width: 100%",
                            value=3,
                            name="step",
                            hx_post=f"/{sid}/steps",
                            hx_target="#steps",
                            hx_swap="outerHTML",
                            hx_trigger="click",
                            hx_on_click=f"window.postMessage({{ type: 'startSession', sessionId: {json.dumps(sid)}, instruction: {json.dumps(instruction)}, task_steps: {json.dumps(task_steps)}, uploadUrl: `${{window.location.origin}}/{upload.__name__}`}})",
                        ),
                    )
                    if num_steps == 3
                    else ""
                ),
            ),
        ),
        Br(),
        Div(
            B("Step 4: "),
            Br(),
            "Send a final answer from the side panel in the new window to finish the study.",
            Br(),
            make_uploader(sid) if num_steps == 4 else "",
            Br(),
            Br(),
            name="step",
            hx_post=f"/{sid}/steps",
            hx_target="#steps",
            hx_swap="outerHTML",
            hx_trigger="change",
        ),
    ]
    return Div(
        *steps[: 2 * num_steps],
        id="steps",
    )


@app.get("/{tid}")
def task_page(
    tid: str,
    pid: Optional[str] = None,
    stid: Optional[str] = None,
    sid: Optional[str] = None,
):
    if tid not in task_configs:
        return Titled("Invalid Task ID", P("Invalid Task ID"))

    if sid is None:
        sid = datetime.now().strftime("%Y%m%d%H%M%S")

    if stid is None:
        stid = "local"

    session_configs[sid] = deepcopy(task_configs[tid])
    session_configs[sid].update(
        dict(
            prolific_id=pid,
            study_id=stid or "local",
            session_id=sid,
            task_id=tid,
        )
    )

    return Container(
        make_steps(sid, 1),
        Script(
            f"""
            window.sessionEvents = [];

            window.addEventListener('message', (event) => {{
                console.log('message event', event);
                if (event.data.type === 'addEvent') {{
                    window.sessionEvents.push(event.data.data);
                    htmx.trigger('#recording-contents', 'addEvent', {{event: event.data.data.event, screenshot: event.data.data.screenshot}})
                }}
            }});
            """
        ),
    )


@app.post("/{sid}/steps")
def create_steps(sid: str, step: int):
    return make_steps(sid, step + 1)


@app.post("/{sid}/update_selector")
def update_selector(sid: str, action: Action):
    return make_task_selector(sid, 0, action.curr_category)


@app.post("/{sid}/update_instruction")
def update_instruction(sid: str, action: Action):
    curr_category = session_configs[sid]["curr_category"]
    return make_task_selector(sid, action.curr_instruction_idx, curr_category)


@app.post("/{sid}/update_feedback")
def update_feedback(sid: str, action: Action):
    session_configs[sid]["feedback"] = action.feedback
    return action.feedback


serve(port=args.port, reload=IS_DEVELOPMENT_ENVIRONMENT)

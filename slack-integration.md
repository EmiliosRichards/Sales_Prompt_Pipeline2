## Slack Integration: End-of-Run Notifications

This document explains how Slack notifications are implemented and triggered at the end of runs in this project, and how to integrate a similar capability into another codebase. It includes code references, configuration, the message payload format, control flow, and extension points. Placeholders for secrets and IDs are provided below.

### What it does
- Posts a message to a Slack channel when reports are generated at the end of a pipeline run.
- Attaches the generated report file (CSV/XLSX) to the message.
- Includes run context: mode/profile, input filename, rows processed, run ID, and command.

### High-level flow
1. Reports are generated near the end of the pipeline in `generate_all_reports(...)`.
2. After report files are written, the code calls `send_slack_notification(...)`.
3. `send_slack_notification(...)` uses the Slack SDK `files_upload_v2` API to upload the file and post an initial message, provided Slack notifications are enabled and credentials are configured.

### Configuration (environment variables)
- `ENABLE_SLACK_NOTIFICATIONS` (bool): Enables Slack posting when `true`.
- `SLACK_BOT_TOKEN` (secret): Bot User OAuth Token (`xoxb-...`).
- `SLACK_CHANNEL_ID` (string): Target channel ID for normal runs (e.g., `CXXXXXX`).
- `SLACK_TEST_CHANNEL_ID` (string): Alternate channel ID used when the app runs in test mode.

These are loaded by `AppConfig`:
```19:23:src/core/config.py
self.enable_slack_notifications: bool = os.getenv('ENABLE_SLACK_NOTIFICATIONS', 'False').lower() == 'true'
self.slack_bot_token: Optional[str] = os.getenv('SLACK_BOT_TOKEN')
self.slack_channel_id: Optional[str] = os.getenv('SLACK_CHANNEL_ID')
self.slack_test_channel_id: Optional[str] = os.getenv('SLACK_TEST_CHANNEL_ID')
```

Test mode is toggled by the CLI flag `--test`, which sets `AppConfig(test_mode=True)` and routes messages to `SLACK_TEST_CHANNEL_ID` when available.

### Where messages are sent from
- Report orchestration triggers Slack uploads after generating the Sales Outreach reports:
```108:121:src/reporting/main_report_orchestrator.py
if sales_outreach_report_path_csv:
    send_slack_notification(
        config=app_config,
        file_path=sales_outreach_report_path_csv,
        report_name="Sales Outreach Report (CSV)",
        run_id=run_id,
        input_file=os.path.basename(original_input_file_path),
        rows_processed=len(all_golden_partner_match_outputs),
        mode=app_config.input_file_profile_name,
        run_command=run_command or ""
    )
```
- Additional reports (e.g., canonical domain summary, row attrition) also attempt to send Slack messages from `src/reporting/report_generator.py` using the same notifier.

### The Slack notifier
Implementation lives in `src/reporting/slack_notifier.py`:
```32:63:src/reporting/slack_notifier.py
if not config.enable_slack_notifications:
    logger.info("Slack notifications are disabled. Skipping.")
    return

channel_id = config.slack_test_channel_id if config.test_mode else config.slack_channel_id

if not all([config.slack_bot_token, channel_id]):
    logger.warning("Slack bot token or channel ID is not configured for the selected mode. Skipping notification.")
    return

client = WebClient(token=config.slack_bot_token)

message = f"""Shop Confirmation System Pipeline Run Complete
---------------------------------------------
Mode: `{mode}`
Input File: `{input_file}`
Rows Processed: {rows_processed}
Run ID: `{run_id}`
Command: `{run_command}`
---------------------------------------------
Report is attached."""

try:
    response = client.files_upload_v2(
        channel=channel_id,
        file=file_path,
        title=os.path.basename(file_path),
        initial_comment=message,
    )
    logger.info(f"Successfully uploaded {report_name} to Slack.")
except SlackApiError as e:
    logger.error(f"Error uploading file to Slack: {e.response['error']}")
```

Notes:
- Uses `slack_sdk` and the `files_upload_v2` endpoint.
- Skips gracefully when disabled or misconfigured.
- Chooses channel based on `test_mode`.

### CLI and test mode
`main_pipeline.py` accepts `-t/--test` to enable test mode. When set, `AppConfig(test_mode=True)` is used, and the notifier posts to `SLACK_TEST_CHANNEL_ID` if present.

### Dependencies
Add to your project:
```1:30:requirements.txt
slack_sdk
```

### How to port this into another project
1. Install `slack_sdk` and ensure your environment can load secrets.
2. Create a config object or equivalent that holds:
   - `enable_slack_notifications: bool`
   - `slack_bot_token: str`
   - `slack_channel_id: str`
   - `slack_test_channel_id: Optional[str]`
   - `test_mode: bool`
3. Add a notifier similar to `send_slack_notification(...)` that:
   - Validates config and picks channel based on `test_mode`.
   - Builds a summary message string with your run context.
   - Calls `WebClient(...).files_upload_v2(...)` with `file`, `title`, and `initial_comment`.
4. Call the notifier after your reports are written, passing:
   - `file_path`, `report_name`, `run_id`, `input_file`, `rows_processed`, `mode`, `run_command`.

### Secrets & IDs placeholders (fill these in)
- SLACK_BOT_TOKEN: [ paste your `xoxb-...` here ]
- SLACK_CHANNEL_ID: [ paste your production channel ID here, e.g., `C12345678` ]
- SLACK_TEST_CHANNEL_ID: [ optional test channel ID, e.g., `C87654321` ]

### Minimal usage example (pseudocode)
```python
from slack_sdk import WebClient

def send_slack_notification(config, file_path, message, title):
    if not (config.enable_slack_notifications and config.slack_bot_token):
        return
    channel = config.slack_test_channel_id if config.test_mode else config.slack_channel_id
    if not channel:
        return
    client = WebClient(token=config.slack_bot_token)
    client.files_upload_v2(channel=channel, file=file_path, title=title, initial_comment=message)
```

### Drop-in, copy/paste implementation (standalone)
Use this snippet directly in another project without needing the files referenced above.

```python
import os
import os.path
from dataclasses import dataclass
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


@dataclass
class SlackConfig:
    enable_slack_notifications: bool = False
    slack_bot_token: Optional[str] = None
    slack_channel_id: Optional[str] = None
    slack_test_channel_id: Optional[str] = None
    test_mode: bool = False

    @staticmethod
    def from_env() -> "SlackConfig":
        return SlackConfig(
            enable_slack_notifications=os.getenv("ENABLE_SLACK_NOTIFICATIONS", "false").lower() == "true",
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
            slack_channel_id=os.getenv("SLACK_CHANNEL_ID"),
            slack_test_channel_id=os.getenv("SLACK_TEST_CHANNEL_ID"),
            # Optional: use APP_TEST_MODE to toggle test routing in non-CLI contexts
            test_mode=os.getenv("APP_TEST_MODE", "false").lower() == "true",
        )


def send_slack_notification(
    config: SlackConfig,
    file_path: str,
    report_name: str,
    run_id: str,
    input_file: str,
    rows_processed: int,
    mode: str,
    run_command: str,
) -> None:
    if not config.enable_slack_notifications:
        return

    channel_id = config.slack_test_channel_id if config.test_mode else config.slack_channel_id
    if not (config.slack_bot_token and channel_id):
        return

    client = WebClient(token=config.slack_bot_token)

    message = (
        "Pipeline Run Complete\n"
        "---------------------------------------------\n"
        f"Mode: `{mode}`\n"
        f"Input File: `{input_file}`\n"
        f"Rows Processed: {rows_processed}\n"
        f"Run ID: `{run_id}`\n"
        f"Command: `{run_command}`\n"
        "---------------------------------------------\n"
        "Report is attached."
    )

    try:
        client.files_upload_v2(
            channel=channel_id,
            file=file_path,
            title=os.path.basename(file_path) if report_name is None else report_name,
            initial_comment=message,
        )
    except SlackApiError as e:
        # Optional: replace with your logging
        print(f"Slack upload error: {getattr(e.response, 'data', e.response).get('error', str(e))}")


if __name__ == "__main__":
    # Fill these or rely on environment variables
    cfg = SlackConfig.from_env()
    # Example placeholders; replace with your real values when calling
    send_slack_notification(
        config=cfg,
        file_path="/path/to/SalesOutreachReport_XXXX.csv",  # <-- your generated report path
        report_name="Sales Outreach Report (CSV)",
        run_id="run_YYYYMMDD_HHMMSS",
        input_file="your_input.csv",
        rows_processed=123,
        mode="final_80k",
        run_command="python main.py -i your_input.csv",
    )
```

### Common pitfalls
- Leaving `ENABLE_SLACK_NOTIFICATIONS=False` or not setting `SLACK_BOT_TOKEN`/`SLACK_CHANNEL_ID` results in silent skips (by design).
- Message header currently says “Shop Confirmation System Pipeline Run Complete”; update to your project name if desired.

### Extension ideas
- Add blocks-rich formatting rather than a plain text initial comment.
- Post a threaded summary without an attachment and upload the file in the thread.
- Include metrics (durations, error summaries) as fields or blocks.
- Add retry/backoff with `tenacity` on transient Slack API failures.



import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from src.core.config import AppConfig

logger = logging.getLogger(__name__)

def send_slack_notification(
    config: AppConfig,
    file_path: str,
    report_name: str,
    run_id: str,
    input_file: str,
    rows_processed: int,
    mode: str,
    run_command: str
):
    """
    Sends a notification with a file to a Slack channel.

    Args:
        config (AppConfig): The application configuration.
        file_path (str): The path to the file to upload.
        report_name (str): The name of the report being sent.
        run_id (str): The ID of the pipeline run.
        input_file (str): The name of the input file.
        rows_processed (int): The number of rows processed.
        mode (str): The processing mode/profile.
        run_command (str): The command used to execute the pipeline.
    """
    if not config.enable_slack_notifications:
        logger.info("Slack notifications are disabled. Skipping.")
        return

    channel_id = config.slack_test_channel_id if config.test_mode else config.slack_channel_id

    if not all([config.slack_bot_token, channel_id]):
        logger.warning("Slack bot token or channel ID is not configured for the selected mode. Skipping notification.")
        return

    client = WebClient(token=config.slack_bot_token)
    
    message = f"""Sales Prompt Pipeline Fullpipe Run Complete
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

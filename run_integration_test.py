import os
import subprocess
import sys
import time
import pandas as pd
import shutil
from dotenv import set_key, get_key

def run_test():
    """
    Runs an integration test of the main pipeline.
    """
    print("--- Starting Integration Test ---")
    env_path = ".env"

    # --- Setup ---
    print("Setting up test environment...")
    
    # Clear the cache directory to ensure a clean run
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        print(f"Clearing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    original_input_path = get_key(env_path, "INPUT_EXCEL_FILE_PATH")
    original_row_range = get_key(env_path, "ROW_PROCESSING_RANGE")
    original_slack_channel = get_key(env_path, "SLACK_CHANNEL_ID")
    original_profile_name = get_key(env_path, "INPUT_FILE_PROFILE_NAME")

    try:
        set_key(env_path, "INPUT_EXCEL_FILE_PATH", "data/test_data.csv")
        set_key(env_path, "INPUT_FILE_PROFILE_NAME", "final_80k")
        set_key(env_path, "LOG_LEVEL", "INFO")
        set_key(env_path, "CONSOLE_LOG_LEVEL", "INFO")
        set_key(env_path, "ROW_PROCESSING_RANGE", "")
        
        slack_channel_to_use = os.getenv("SLACK_TEST_CHANNEL_ID")
        if not slack_channel_to_use:
            slack_channel_to_use = original_slack_channel

        if slack_channel_to_use is not None:
            set_key(env_path, "SLACK_CHANNEL_ID", slack_channel_to_use)

        # --- Execution ---
        print("Running main pipeline...")
        process = subprocess.Popen([sys.executable, "main_pipeline.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        print("--- Pipeline Output ---")
        print(stdout.decode('latin-1'))
        if stderr:
            print("--- Errors ---")
            print(stderr.decode('latin-1'))
        print("--- End Pipeline Output ---")

        # --- Validation ---
        print("Validating test results...")
        output_dir = max([os.path.join("output_data", d) for d in os.listdir("output_data") if os.path.isdir(os.path.join("output_data", d))], key=os.path.getmtime)
        
        sales_report_files = [f for f in os.listdir(output_dir) if f.startswith("SalesOutreachReport")]
        failed_rows_files = [f for f in os.listdir(output_dir) if f.startswith("failed_rows")]

        if not sales_report_files:
            raise FileNotFoundError("Sales Outreach Report not found!")
        if not failed_rows_files:
            raise FileNotFoundError("Failed Rows Report not found!")

        sales_report_path = os.path.join(output_dir, sales_report_files[0])
        failed_rows_path = os.path.join(output_dir, failed_rows_files[0])

        sales_report = pd.read_csv(sales_report_path)
        failed_rows = pd.read_csv(failed_rows_path)

        # --- Assertions ---
        print("Running assertions...")
        
        def check_assertion(condition, message):
            if not condition:
                print(f"❌ {message}")
                return False
            print(f"✅ {message}")
            return True

        all_tests_passed = True
        all_tests_passed &= check_assertion("HubSpot" in sales_report["Company Name"].values, "HubSpot in sales report")
        all_tests_passed &= check_assertion("Nike" in failed_rows["CompanyName"].values, "Nike in failed rows")
        all_tests_passed &= check_assertion("Small Biz LLC" in failed_rows["CompanyName"].values, "Small Biz LLC in failed rows")
        all_tests_passed &= check_assertion("Invalid URL Ltd." in failed_rows["CompanyName"].values, "Invalid URL Ltd. in failed rows")
        all_tests_passed &= check_assertion("Scraper Fail Co." in failed_rows["CompanyName"].values, "Scraper Fail Co. in failed rows")
        all_tests_passed &= check_assertion("No Text Corp." in failed_rows["CompanyName"].values, "No Text Corp. in failed rows")
        all_tests_passed &= check_assertion("Scraper Fallback Inc." in sales_report["Company Name"].values, "Scraper Fallback Inc. in sales report")
        
        if "Scraper Fallback Inc." in sales_report["Company Name"].values:
            all_tests_passed &= check_assertion(sales_report[sales_report["Company Name"] == "Scraper Fallback Inc."]["ScrapeStatus"].iloc[0] == "Used_Fallback_Description", "Scraper Fallback Inc. used fallback description")

        all_tests_passed &= check_assertion("Exxomove" in sales_report["Company Name"].values, "Exxomove in sales report")
        if "Exxomove" in sales_report["Company Name"].values:
            all_tests_passed &= check_assertion(pd.notna(sales_report[sales_report["Company Name"] == "Exxomove"]["found_number"].iloc[0]), "Exxomove has a found phone number")

        if all_tests_passed:
            print("\n🎉 All tests passed! 🎉")
        else:
            print("\n🔥 Some tests failed. Please review the output above. 🔥")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        sys.exit(1)
    finally:
        # --- Cleanup ---
        print("Restoring original environment...")
        if original_input_path is not None:
            set_key(env_path, "INPUT_EXCEL_FILE_PATH", original_input_path)
        else:
            set_key(env_path, "INPUT_EXCEL_FILE_PATH", "")
            
        if original_row_range is not None:
            set_key(env_path, "ROW_PROCESSING_RANGE", original_row_range)
        else:
            set_key(env_path, "ROW_PROCESSING_RANGE", "")
            
        if original_slack_channel is not None:
            set_key(env_path, "SLACK_CHANNEL_ID", original_slack_channel)
        else:
            set_key(env_path, "SLACK_CHANNEL_ID", "")
            
        if original_profile_name is not None:
            set_key(env_path, "INPUT_FILE_PROFILE_NAME", original_profile_name)
        else:
            set_key(env_path, "INPUT_FILE_PROFILE_NAME", "default")

if __name__ == "__main__":
    run_test()
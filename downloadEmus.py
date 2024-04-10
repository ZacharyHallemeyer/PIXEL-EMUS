from emm_api import emm_search_and_download
from datetime import datetime, timedelta

def download_emu_l2a_files(download_dir):
    instrument = "emu"
    level = "l2a"
    start_date_str = "2021-02-01"
    end_date_str = "2021-03-31"
    latest = 'true'

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current_date = start_date

    while current_date <= end_date:
        start_date_str = current_date.strftime("%Y-%m-%d")
        end_date_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        emm_search_and_download(download_dir=download_dir,
                                instrument=instrument,
                                level=level,
                                latest=latest,
                                start_date=start_date_str,
                                end_date=end_date_str,
                                use_sdc_dir_structure=True)
        current_date += timedelta(days=1)


download_emu_l2a_files('../Data')

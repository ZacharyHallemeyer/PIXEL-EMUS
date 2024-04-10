from emm_api import emm_search_and_download

def download_emu_l2a_files(download_dir):

    instrument = "emu"
    level = "l2a"
    start_date = "2021-02-01"
    end_date = "2021-03-31"
    latest = 'true'

    emm_search_and_download(download_dir=download_dir,
                            instrument=instrument,
                            level=level,
                            latest=latest,
                            start_date=start_date,
                            end_date=end_date,
                            use_sdc_dir_structure=True)

download_emu_l2a_files('../Data')

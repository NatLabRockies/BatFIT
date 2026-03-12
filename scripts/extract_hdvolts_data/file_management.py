from batfit import logger
import os
import sys


def get_cycling_protocol_folder(cycling_protocol: str) -> str:
    """
    Get cycling protocol folder from input
    """

    if "1" in cycling_protocol:
        cycling_protocol_id = 1
    elif "2" in cycling_protocol:
        cycling_protocol_id = 2
    elif "3" in cycling_protocol:
        cycling_protocol_id = 3
    elif "4" in cycling_protocol:
        cycling_protocol_id = 4
    else:
        raise NotImplementedError

    if "lh" in cycling_protocol.lower():
        cycling_protocol_type = "LH"
    elif "rh" in cycling_protocol.lower():
        cycling_protocol_type = "RH"
    else:
        raise NotImplementedError

    cycling_protocol_folder = f"{cycling_protocol_type}-{cycling_protocol_id}"
    assert cycling_protocol_folder in [
        "LH-1",
        "LH-2",
        "LH-3",
        "LH-4",
        "RH-1",
        "RH-2",
        "RH-3",
        "RH-4",
    ]
    return cycling_protocol_folder


def get_available_rpt_folder(
    cycling_protocol_folder: str,
) -> Tuple(list[str], list[id]):
    """
    Get the list of RPT folders available
    """
    if not os.path.isabs(cycling_protocol_folder):
        logger.error(f"{cycling_protocol_folder} is not absolute")
        sys.exit()
    list_folder = os.listdir(cycling_protocol_folder)
    available_rpt_folder = []
    available_rpt_id = []
    if "BOL" in list_folder:
        available_rpt_folder.append("BOL")
        available_rpt_id.append(-1)
    available_rpt_folder += [
        folder for folder in list_folder if folder.startswith("RPT")
    ]
    available_rpt_id += [
        int(folder[3:]) for folder in list_folder if folder.startswith("RPT")
    ]
    return available_rpt_folder, available_rpt_id


def get_rpt_folder(rpt_id: int, available_rpt_folder: list[str]) -> str:
    """
    Get RPT folder from input
    """
    if rpt_id < 0:
        rpt_folder = "BOL"
    else:
        rpt_folder = f"RPT{rpt_id}"
    if rpt_folder in available_rpt_folder:
        return rpt_folder
    else:
        logger.error(f"{rpt_folder} not in {available_rpt_folder}")
        sys.exit()


def get_data_folder(
    cycling_protocol: str,
    rpt_id: int,
    data_type: str = "diffcap",
    data_root: str = "/Users/mhassana/Desktop/GitHub/HDVOLTS_data",
) -> str:
    """
    Get data folder from input
    """
    assert data_type.lower() in ["hppc", "diffcap", "cycle"]

    cycling_protocol_folder = get_cycling_protocol_folder(cycling_protocol)
    cycling_protocol_folder = os.path.join(data_root, cycling_protocol_folder)
    logger.debug(
        f"Cycling protocol folder : {cycling_protocol_folder} deduced from {cycling_protocol}"
    )
    rpt_folder = get_rpt_folder(
        rpt_id, get_available_rpt_folder(cycling_protocol_folder)[0]
    )
    logger.debug(f"RPT folder : {rpt_folder} deduced from {rpt_id}")

    if data_type.lower() == "hppc":
        data_folder = os.path.join(cycling_protocol_folder, rpt_folder, "HPPC")
    elif data_type.lower() == "diffcap":
        data_folder = os.path.join(
            cycling_protocol_folder, rpt_folder, "DiffCap"
        )
        if not os.path.isdir(data_folder):
            data_folder = os.path.join(
                cycling_protocol_folder, rpt_folder, "DifCap"
            )
        if not os.path.isdir(data_folder):
            logger.error(f"Could not find {data_folder}")
            sys.exit()
    elif data_type.lower() == "cycle":
        list_folder = os.listdir(
            os.path.join(cycling_protocol_folder, rpt_folder)
        )
        for folder in list_folder:
            if "Cycle" in folder:
                break
        data_folder = os.path.join(cycling_protocol_folder, rpt_folder, folder)

    logger.debug(f"Data Folder : {data_folder}")
    return data_folder


def cells_protocols_pairs():
    # Only include cells that are consistently observed in HPPC, Diffcap, *Cycle*
    pairs = {
        "LH-2": [17],
        "LH-3": [19, 20, 21],
        "LH-4": [22, 23],
        "RH-1": [1],
        "RH-2": [4, 5, 6],
        "RH-3": [7, 8, 9],
    }
    return pairs


def format_cell_name(cell_id: int) -> str:
    return f"{cell_id:03}"


def get_data_file(
    cycling_protocol: str,
    rpt_id: int,
    data_type: str,
    data_root: str = "/Users/mhassana/Desktop/GitHub/HDVOLTS_data",
) -> dict:
    """
    Returns a dict [cellid: cell file]
    """
    data_folder = get_data_folder(
        cycling_protocol, rpt_id, data_type, data_root
    )
    cell_prot_pairs = cells_protocols_pairs()
    protocol_folder = get_cycling_protocol_folder(cycling_protocol)
    cells = cell_prot_pairs[protocol_folder]

    list_files = os.listdir(data_folder)
    file_keep = {}
    for cell in cells:
        for file in list_files:
            if file.endswith(format_cell_name(cell) + ".csv"):
                file_keep[cell] = os.path.join(data_folder, file)
    return file_keep


def get_all_rpt_data_file(
    cycling_protocol: str,
    data_type: str,
    data_root: str = "/Users/mhassana/Desktop/GitHub/HDVOLTS_data",
) -> dict:
    cycling_protocol_folder = get_cycling_protocol_folder(cycling_protocol)
    cycling_protocol_folder = os.path.join(data_root, cycling_protocol_folder)
    available_rpt_folder, available_rpt_id = get_available_rpt_folder(
        cycling_protocol_folder
    )

    data = {}
    for rpt_id in available_rpt_id:
        data[rpt_id] = get_data_file(
            cycling_protocol, rpt_id, data_type, data_root=data_root
        )

    return data


def get_all_protocol_all_rpt_data_file(
    data_type: str,
    data_root: str = "/Users/mhassana/Desktop/GitHub/HDVOLTS_data",
) -> dict:
    cycling_protocol_list = list(cells_protocols_pairs().keys())
    data = {}
    for cycling_protocol in cycling_protocol_list:
        data[cycling_protocol] = get_all_rpt_data_file(
            cycling_protocol, data_type, data_root
        )
    return data
